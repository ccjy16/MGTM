from math import ceil
import ast
import os
import logging
import json
from pathlib import Path
from PIL import Image, ImageFile
import base64
from io import BytesIO
from dataclasses import dataclass
import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
DATA_TYPE = "intent_style_attribute"

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


def parse_sample_record(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        record = ast.literal_eval(line)
        if not isinstance(record, dict):
            raise ValueError("MultiChat sample metadata must be a mapping")
        return record


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    return text.lower()


def build_image_transform(split, use_augment, resolution):
    operations = [
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
    if split == "train" and use_augment:
        operations.insert(0, RandomHorizontalFlip())
    return Compose(operations)


def _resolve_data_mode(lmdb_path, mode_name=None):
    if mode_name and mode_name not in {"all", "total"}:
        return mode_name
    if mode_name == "all":
        return "all"
    parent_name = Path(lmdb_path).parent.name
    if parent_name.startswith("lmdb_"):
        parent_name = parent_name[len("lmdb_"):]
    if parent_name.endswith("_intent_style_attribute"):
        parent_name = parent_name[:-len("_intent_style_attribute")]
    return parent_name or "all"


def _load_intent_mappings(multichat_root, mode_name):
    intent_mode = "all" if mode_name in {"all", "total"} else mode_name
    intent_path = os.path.join(multichat_root, intent_mode, "intent.txt")
    if not os.path.isfile(intent_path):
        raise FileNotFoundError(
            f"Intent mapping for data_mode={mode_name!r} was not found: {intent_path}"
        )

    id2intent = {}
    intent2id = {}
    with open(intent_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split("\t", 1)
            if len(fields) != 2:
                continue
            intent_id = int(fields[0])
            intent_name = fields[1].strip().lower()
            if intent_name in intent2id:
                logger.warning(
                    "Duplicate intent name %r in %s; keeping ID %s and ignoring ID %s",
                    intent_name, intent_path, intent2id[intent_name], intent_id,
                )
                continue
            id2intent[intent_id] = intent_name
            intent2id[intent_name] = intent_id
    return id2intent, intent2id


def _lookup_intent_id(intent2id, intent_name, mode_name):
    normalized_name = str(intent_name).lower()
    if normalized_name not in intent2id:
        raise ValueError(
            f"Intent {intent_name!r} is not defined for data_mode={mode_name!r}"
        )
    return intent2id[normalized_name]


def get_all_features(lmdb_path, split_mode, use_augment=False, resolution=224,
                     data_mode_name=None):
    logger.info("Loading MultiChat %s split from %s", split_mode, lmdb_path)
    assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split_mode)
    lmdb_pairs = os.path.join(lmdb_path, "pairs")
    assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist! ".format(
        lmdb_pairs, split_mode)
    lmdb_base = os.path.dirname(lmdb_path)  # e.g. .../lmdb_boba_intent_style_attribute
    lmdb_imgs_1 = os.path.join(lmdb_base, split_mode + "_imgs")

    logger.info("Loading MultiChat image LMDB from %s", lmdb_imgs_1)
    assert os.path.isdir(lmdb_imgs_1), f"The LMDB image directory {lmdb_imgs_1} does not exist!"

    env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_pairs = env_pairs.begin(buffers=True)

    env_imgs_1 = lmdb.open(lmdb_imgs_1, readonly=True, create=False, lock=False, readahead=False,
                           meminit=False)
    txn_imgs_1 = env_imgs_1.begin(buffers=True)

    number_samples = int(txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))

    img_id_list_1 = []
    cursor = env_imgs_1.begin().cursor()
    for key, value in cursor:
        if key.decode('utf-8') != 'num_images':
            img_id_list_1.append(key.decode('utf-8'))

    multichat_root = str(Path(lmdb_path).resolve().parents[1])
    mode_name = _resolve_data_mode(lmdb_path, data_mode_name)
    id2intent, intent2id = _load_intent_mappings(multichat_root, mode_name)

    imgid2intent = {}
    mode_list = (
        ['boba', 'kuaile', 'quanguo', 'yongyuan', 'siban']
        if mode_name in {"all", "total"}
        else [mode_name]
    )
    for mo in mode_list:
        json_path = os.path.join(multichat_root, f'{mo}_sample_{DATA_TYPE}.json')
        if os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                datas = f.readlines()
                f.close()
            logger.info("Loading MultiChat sample metadata from %s", json_path)
            for data in datas:
                data = parse_sample_record(data)
                image_id = data['image_ids']
                if image_id not in imgid2intent:
                    imgid2intent[image_id] = []
                intent_id = _lookup_intent_id(
                    intent2id, data['fined_intent'], mode_name
                )
                if intent_id not in imgid2intent[image_id]:
                    imgid2intent[image_id].append(intent_id)

    session_ids = []
    sample_ids = []
    img_ids, contexts, speakers, texts, fined_intents, summaries = {}, {}, {}, {}, {}, {}

    e_transform = build_image_transform(split_mode, use_augment, resolution)

    image_features = []

    error_count = 0
    total_count = 0

    for i in range(number_samples):
        pair = pickle.loads(txn_pairs.get("{}".format(i).encode('utf-8')).tobytes())
        session_id, sample_id, summary, history, context, speaker, text, fined_intent, image_id = pair

        session_ids.append(session_id)
        sample_ids.append(sample_id)
        img_ids[sample_id] = image_id
        contexts[sample_id] = context
        speakers[sample_id] = speaker
        texts[sample_id] = text
        summaries[sample_id] = summary

        intent_id = _lookup_intent_id(intent2id, fined_intent, mode_name)
        fined_intents[sample_id] = intent_id

        try:
            image_b64 = txn_imgs_1.get("{}".format(image_id).encode('utf-8')).tobytes()
            image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
            image_bytes = base64.urlsafe_b64decode(image_b64)

            if len(image_bytes) < 100:
                raise ValueError(f"Image data too small: {len(image_bytes)} bytes")

            image = Image.open(BytesIO(image_bytes))
            image.load()
            image = e_transform(image)
            image_features.append(image)
        except Exception as e:
            error_count += 1
            logger.warning(f"Failed to load image {image_id} (sample {i}): {str(e)}")
            default_image = Image.new('RGB', (resolution, resolution), color=(128, 128, 128))
            image = e_transform(default_image)
            image_features.append(image)

        total_count += 1
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{number_samples} images, errors: {error_count}")

    logger.info(f"Total images processed: {total_count}, errors: {error_count}, error rate: {error_count/total_count:.2%}")

    all_image_features = {}

    error_count_all = 0
    for i in range(len(img_id_list_1)):
        try:
            image_b64 = txn_imgs_1.get("{}".format(img_id_list_1[i]).encode('utf-8')).tobytes()
            image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
            image_bytes = base64.urlsafe_b64decode(image_b64)

            if len(image_bytes) < 100:
                raise ValueError(f"Image data too small: {len(image_bytes)} bytes")

            image = Image.open(BytesIO(image_bytes))
            image.load()
            image = e_transform(image)
            all_image_features[img_id_list_1[i]] = image
        except Exception as e:
            error_count_all += 1
            logger.warning(f"Failed to load image {img_id_list_1[i]} in all_image_features: {str(e)}")
            default_image = Image.new('RGB', (resolution, resolution), color=(128, 128, 128))
            image = e_transform(default_image)
            all_image_features[img_id_list_1[i]] = image

    logger.info(f"All images processed: {len(img_id_list_1)}, errors: {error_count_all}")

    return (
        session_ids, sample_ids, img_ids, contexts, speakers,
        texts, fined_intents, summaries, image_features, number_samples,
        all_image_features, imgid2intent, id2intent,
    )


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False,
                 resolution=224, text_feature_mode="cn_clip", data_mode_name=None):
        lmdb_path = lmdb_path
        logger.info("Initializing MultiChat %s dataset from %s", split, lmdb_path)
        self.max_txt_length = max_txt_length
        (
            self.session_ids, self.sample_ids, self.img_ids,
            self.contexts, self.speakers, self.texts, self.fined_intents,
            self.summaries, self.image_features, self.number_samples,
            self.all_image_features, self.intent2id, self.id2intent,
        ) = get_all_features(
                lmdb_path, split, use_augment=use_augment, resolution=resolution,
                data_mode_name=data_mode_name)
        self.dataset_len = self.number_samples
        self.split = split
        self.use_augment = use_augment
        self.text_feature_mode = text_feature_mode
        self.raw_text_strings = self._build_raw_text_strings()
        if self.text_feature_mode != "taiyi_CLIP":
            raise ValueError("MGTM supports only the taiyi_CLIP text tower")

    def _build_transform(self, resolution):
        return build_image_transform(self.split, self.use_augment, resolution)

    def _build_raw_text_strings(self):
        raw_text_strings = {}
        for sample_id, context in self.contexts.items():
            raw_text = context.split('\t')
            raw_text = '[SEP]'.join(raw_text)
            raw_text_strings[sample_id] = _preprocess_text(raw_text)
        return raw_text_strings

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples
        session_id = self.session_ids[sample_index]
        sample_id = self.sample_ids[sample_index]
        image_feature = self.image_features[sample_index]

        image_id = self.img_ids[sample_id]
        text = self.texts[sample_id]
        fined_intent = self.fined_intents[sample_id]
        summary = self.summaries[sample_id]
        image = image_feature

        raw_text_input = self.raw_text_strings[sample_id]

        session_id_tensor = torch.tensor(int(session_id), dtype=torch.long)

        return session_id_tensor, image_id, image, raw_text_input, \
            fined_intent, summary


def pad_dataset(dataset, batch_size):
    dataset.dataset_len = ceil(dataset.dataset_len / batch_size) * batch_size
    dataset.batch_size = batch_size


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: Sampler
    dataset: LMDBDataset
    epoch_id: int


class SessionSummaryPool:

    def __init__(self):
        self.session_ids = []
        self.summaries = {}         # session_id -> summary text
        self.embeddings = {}        # session_id -> embedding tensor (cpu)
        self._tensor_cache = {}

    def add_session(self, session_id, summary_text, embedding=None):
        sid = int(session_id)
        if sid not in self.summaries:
            self.session_ids.append(sid)
            self.summaries[sid] = summary_text
            if embedding is not None:
                self.embeddings[sid] = embedding
            self._tensor_cache.clear()

    def get_unique_sessions_from_dataset(self, dataset):
        seen = set()
        for i in range(dataset.number_samples):
            sid = int(dataset.session_ids[i])
            if sid not in seen:
                seen.add(sid)
                sample_id = dataset.sample_ids[i]
                summary = dataset.summaries[sample_id]
                self.add_session(sid, summary)

    def refresh_online_cache(self, model, device, encode_batch_size=64):
        if encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be greater than zero")

        sorted_ids = sorted(self.session_ids)
        refreshed = {}
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(sorted_ids), encode_batch_size):
                    chunk_ids = sorted_ids[start:start + encode_batch_size]
                    texts = [self.summaries[sid] for sid in chunk_ids]
                    _, features = model.encode_text(texts, return_pooled=True)
                    for sid, feature in zip(chunk_ids, features):
                        refreshed[sid] = feature.detach().to("cpu", dtype=torch.float32)
        finally:
            model.train(was_training)

        self.embeddings = refreshed
        self._tensor_cache.clear()
        return len(refreshed)

    def _get_cached_tensors(self, device):
        cache_key = str(device)
        sorted_ids = tuple(sorted(self.embeddings.keys()))
        cached = self._tensor_cache.get(cache_key)
        if cached is not None and cached["sorted_ids"] == sorted_ids:
            return cached["sorted_ids"], cached["all_emb"], cached["all_sid"]

        all_emb_list = [self.embeddings[sid].to(device) for sid in sorted_ids]
        all_emb = torch.stack(all_emb_list)
        all_sid = torch.tensor(sorted_ids, device=device)
        self._tensor_cache[cache_key] = {
            "sorted_ids": sorted_ids,
            "all_emb": all_emb,
            "all_sid": all_sid,
        }
        return sorted_ids, all_emb, all_sid

    def query(
        self,
        current_session_ids,
        device,
        max_pool_size=256,
    ):
        B = current_session_ids.shape[0]

        if len(self.embeddings) == 0:
            return None, None, None

        sorted_ids, all_emb, all_sid = self._get_cached_tensors(device)
        H = all_emb.shape[1]

        P = min(len(sorted_ids), max_pool_size)

        pool_emb = torch.zeros(B, P, H, device=device)
        pool_sid = torch.zeros(B, P, dtype=torch.long, device=device)
        pool_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

        for i in range(B):
            curr = current_session_ids[i].item()
            causal_mask = all_sid < curr
            valid = torch.where(causal_mask)[0]
            if len(valid) > 0:
                selected = valid[-P:]
                k = len(selected)
                pool_emb[i, :k] = all_emb[selected]
                pool_sid[i, :k] = all_sid[selected]
                pool_mask[i, :k] = True

        if not pool_mask.any():
            return None, None, None

        return pool_emb, pool_sid, pool_mask

    def get_max_session_id(self):
        return max(self.session_ids) if self.session_ids else 0

    def get_min_session_id(self):
        return min(self.session_ids) if self.session_ids else 0


def custom_collate_fn(batch):
    session_ids, image_ids, images, raw_texts, fined_intents, summaries = zip(*batch)
    session_ids = torch.stack(session_ids, dim=0)
    images = torch.stack(images, dim=0)
    if isinstance(raw_texts[0], torch.Tensor):
        raw_texts = torch.stack(raw_texts, dim=0)
    else:
        raw_texts = list(raw_texts)

    if isinstance(fined_intents[0], str):
        fined_intents = torch.tensor([int(x) for x in fined_intents], dtype=torch.long)
    else:
        fined_intents = torch.tensor(fined_intents, dtype=torch.long)

    return session_ids, image_ids, images, raw_texts, \
        fined_intents, summaries


def select_dataloader_settings(args, is_train):
    training = is_train == 1
    batch_size = args.batch_size if training else args.valid_batch_size
    num_workers = args.num_workers if training else args.valid_num_workers
    use_augment = args.use_augment if training else False
    return batch_size, num_workers, use_augment


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train == 1:
        db_path = args.train_data
        split = 'train'
    elif is_train == 2:
        db_path = args.val_data
        split = 'val'
    elif is_train == 0:
        db_path = args.test_data
        split = 'test'
    assert db_path is not None
    batch_size, num_workers, use_augment = select_dataloader_settings(args, is_train)
    dataset = LMDBDataset(
        db_path,
        split=split,
        max_txt_length=max_txt_length,
        use_augment=use_augment,
        resolution=args.resolution,
        text_feature_mode=args.text_feature_mode,
        data_mode_name=args.data_mode,
    )
    if is_train == 1:
        pad_dataset(dataset, batch_size)
        num_samples = dataset.dataset_len
    else:
        num_samples = dataset.number_samples

    if is_train == 1:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )

    dataloader.num_samples = num_samples
    if is_train == 1:
        assert num_samples % batch_size == 0
    dataloader.num_batches = ceil(num_samples / batch_size)

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}
    if args.train_data:
        data["train"] = get_dataset(
            args,
            is_train=1,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)
    if args.val_data:
        data["val"] = get_dataset(
            args,
            is_train=2,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)
    if args.test_data:
        data["test"] = get_dataset(
            args,
            is_train=0,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)

    return data

import re
from typing import Iterable

ALFWORLD_TASK_TYPES = (
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "clean",
    "heat",
    "cool",
)

WEBSHOP_TASK_TYPES = (
    "apparel",
    "footwear",
    "home_decor",
    "electronics",
    "accessories",
    "beauty_health",
    "other",
)

ALFWORLD_DATASET_TO_TASK_TYPE = {
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
}

WEBSHOP_KEYWORDS = (
    (
        "apparel",
        (
            "shirt", "dress", "t-shirt", "polo", "pants", "jeans", "jacket",
            "coat", "sweater", "blouse", "skirt", "shorts", "underwear",
            "swimsuit", "swimwear", "hoodie", "vest", "cardigan", "suit",
            "blazer", "tee", "top", "leggings", "clothing", "clothes",
        ),
    ),
    (
        "footwear",
        (
            "shoe", "shoes", "boot", "boots", "sandal", "sandals", "sneaker",
            "sneakers", "slipper", "slippers", "loafer", "heel", "heels",
            "flat", "flats", "oxford", "pump", "moccasin", "flip-flop",
            "footwear",
        ),
    ),
    (
        "home_decor",
        (
            "pillow", "curtain", "rug", "mat", "blanket", "bedding", "towel",
            "lamp", "decor", "furniture", "cushion", "sheet", "tablecloth",
            "vase", "desk", "table", "chair", "sofa", "candle", "quilt",
        ),
    ),
    (
        "electronics",
        (
            "phone", "laptop", "tablet", "computer", "headphone", "headphones",
            "earphone", "earbud", "earbuds", "speaker", "charger", "cable",
            "mouse", "keyboard", "monitor", "camera", "smartwatch", "battery",
            "electronic", "electronics", "device", "gadget", "adapter", "usb",
        ),
    ),
    (
        "accessories",
        (
            "bag", "wallet", "belt", "hat", "cap", "scarf", "glove", "gloves",
            "jewelry", "necklace", "bracelet", "ring", "earring", "earrings",
            "sunglasses", "glasses", "watch", "purse", "backpack", "handbag",
            "tie", "bow",
        ),
    ),
    (
        "beauty_health",
        (
            "makeup", "cosmetic", "cosmetics", "skincare", "lotion", "cream",
            "shampoo", "conditioner", "perfume", "cologne", "brush", "soap",
            "body wash", "nail", "lipstick", "mascara", "foundation", "serum",
            "moisturizer", "vitamin", "supplement", "health", "toothbrush",
        ),
    ),
)


def _contains_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])",
            text,
        )
        for phrase in phrases
    )


def normalize_benchmark(value: str) -> str:
    value = str(value or "").lower()
    if "webshop" in value:
        return "webshop"
    if "alfworld" in value or "alfred" in value:
        return "alfworld"
    raise ValueError(f"Unsupported benchmark: {value or '<empty>'}")


def task_types_for_benchmark(benchmark: str) -> tuple[str, ...]:
    benchmark = normalize_benchmark(benchmark)
    return ALFWORLD_TASK_TYPES if benchmark == "alfworld" else WEBSHOP_TASK_TYPES


def canonicalize_alfworld_task_type(task_type: str) -> str:
    task_type = str(task_type or "")
    if task_type == "examine":
        return "look_at_obj_in_light"
    if task_type in ALFWORLD_DATASET_TO_TASK_TYPE:
        return ALFWORLD_DATASET_TO_TASK_TYPE[task_type]
    if task_type in ALFWORLD_TASK_TYPES:
        return task_type
    return "unknown"


def classify_alfworld_task(text: str) -> str:
    lowered = str(text or "").lower()
    for raw_type, canonical in ALFWORLD_DATASET_TO_TASK_TYPE.items():
        if raw_type in lowered:
            return canonical
    if _contains_phrase(lowered, ("two", "2")):
        return "pick_two_obj_and_place"
    if _contains_phrase(lowered, ("look at", "examine")):
        return "look_at_obj_in_light"
    if _contains_phrase(lowered, ("clean", "washed", "wash")):
        return "clean"
    if _contains_phrase(lowered, ("heat", "heated", "hot")):
        return "heat"
    if _contains_phrase(lowered, ("cool", "cooled", "cold")):
        return "cool"
    return "pick_and_place"


def classify_webshop_task(text: str) -> str:
    lowered = str(text or "").lower()
    for task_type, keywords in WEBSHOP_KEYWORDS:
        if _contains_phrase(lowered, keywords):
            return task_type
    return "other"


def classify_task(benchmark: str, text: str) -> str:
    benchmark = normalize_benchmark(benchmark)
    match = re.search(
        r"Your task is to:\s*(.*?)(?=\n(?:\n## Retrieved|Prior to|Your current))",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = match.group(1).strip().rstrip(".")
    if benchmark == "alfworld":
        return classify_alfworld_task(text)
    return classify_webshop_task(text)

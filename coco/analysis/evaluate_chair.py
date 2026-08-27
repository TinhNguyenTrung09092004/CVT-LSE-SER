# %%
# !pip install -q pandas numpy nltk

# %%
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# %%
# Vendored verbatim from CLiPS pattern (pattern/text/en/inflect.py, BSD-3),
# itself adapted from Bermi Ferrer's Inflector for Python (MIT).
# `from pattern.en import singularize` is not installable on Python 3.12.

NOUN = "NN"

plural_prepositions = set((
    "about"  , "before" , "during", "of"   , "till" ,
    "above"  , "behind" , "except", "off"  , "to"   ,
    "across" , "below"  , "for"   , "on"   , "under",
    "after"  , "beneath", "from"  , "onto" , "until",
    "among"  , "beside" , "in"    , "out"  , "unto" ,
    "around" , "besides", "into"  , "over" , "upon" ,
    "at"     , "between", "near"  , "since", "with" ,
    "athwart", "betwixt",
               "beyond",
               "but",
               "by"))

singular_rules = [
    (r'(?i)(.)ae$'            , '\\1a'    ),
    (r'(?i)(.)itis$'          , '\\1itis' ),
    (r'(?i)(.)eaux$'          , '\\1eau'  ),
    (r'(?i)(quiz)zes$'        , '\\1'     ),
    (r'(?i)(matr)ices$'       , '\\1ix'   ),
    (r'(?i)(ap|vert|ind)ices$', '\\1ex'   ),
    (r'(?i)^(ox)en'           , '\\1'     ),
    (r'(?i)(alias|status)es$' , '\\1'     ),
    (r'(?i)([octop|vir])i$'   , '\\1us'  ),
    (r'(?i)(cris|ax|test)es$' , '\\1is'   ),
    (r'(?i)(shoe)s$'          , '\\1'     ),
    (r'(?i)(o)es$'            , '\\1'     ),
    (r'(?i)(bus)es$'          , '\\1'     ),
    (r'(?i)([m|l])ice$'       , '\\1ouse' ),
    (r'(?i)(x|ch|ss|sh)es$'   , '\\1'     ),
    (r'(?i)(m)ovies$'         , '\\1ovie' ),
    (r'(?i)(.)ombies$'        , '\\1ombie'),
    (r'(?i)(s)eries$'         , '\\1eries'),
    (r'(?i)([^aeiouy]|qu)ies$', '\\1y'    ),
    (r"([aeo]l)ves$"          , "\\1f"    ),
    (r"([^d]ea)ves$"          , "\\1f"    ),
    (r"arves$"                , "arf"     ),
    (r"erves$"                , "erve"    ),
    (r"([nlw]i)ves$"          , "\\1fe"   ),
    (r'(?i)([lr])ves$'        , '\\1f'    ),
    (r"([aeo])ves$"           , "\\1ve"   ),
    (r'(?i)(sive)s$'          , '\\1'     ),
    (r'(?i)(tive)s$'          , '\\1'     ),
    (r'(?i)(hive)s$'          , '\\1'     ),
    (r'(?i)([^f])ves$'        , '\\1fe'   ),
    (r'(?i)(^analy)ses$'      , '\\1sis'  ),
    (r'(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\\1\\2sis'),
    (r'(?i)(.)opses$'         , '\\1opsis'),
    (r'(?i)(.)yses$'          , '\\1ysis' ),
    (r'(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'),
    (r'(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$', '\\1ose'),
    (r'(?i)(.)oses$'          , '\\1osis' ),
    (r'(?i)([ti])a$'          , '\\1um'   ),
    (r'(?i)(n)ews$'           , '\\1ews'  ),
    (r'(?i)s$'                , ''        ),
]

singular_rules = [(re.compile(r[0]), r[1]) for r in singular_rules]

singular_uninflected = set((
    "bison"      , "debris"   , "headquarters", "pincers"    , "trout"     ,
    "bream"      , "diabetes" , "herpes"      , "pliers"     , "tuna"      ,
    "breeches"   , "djinn"    , "high-jinks"  , "proceedings", "whiting"   ,
    "britches"   , "eland"    , "homework"    , "rabies"     , "wildebeest",
    "carp"       , "elk"      , "innings"     , "salmon"     ,
    "chassis"    , "flounder" , "jackanapes"  , "scissors"   ,
    "christmas"  , "gallows"  , "mackerel"    , "series"     ,
    "clippers"   , "georgia"  , "measles"     , "shears"     ,
    "cod"        , "graffiti" , "mews"        , "species"    ,
    "contretemps",              "mumps"       , "swine"      ,
    "corps"      ,              "news"        , "swiss"      ,
))
singular_uncountable = set((
    "advice"     , "equipment", "happiness"   , "luggage"    , "news"      , "software"     ,
    "bread"      , "fruit"    , "information" , "mathematics", "progress"  , "understanding",
    "butter"     , "furniture", "ketchup"     , "mayonnaise" , "research"  , "water"        ,
    "cheese"     , "garbage"  , "knowledge"   , "meat"       , "rice"      ,
    "electricity", "gravel"   , "love"        , "mustard"    , "sand"      ,
))
singular_ie = set((
    "alergie"    , "cutie"    , "hoagie"      , "newbie"     , "softie"    , "veggie"       ,
    "auntie"     , "doggie"   , "hottie"      , "nightie"    , "sortie"    , "weenie"       ,
    "beanie"     , "eyrie"    , "indie"       , "oldie"      , "stoolie"   , "yuppie"       ,
    "birdie"     , "freebie"  , "junkie"      , "^pie"       , "sweetie"   , "zombie"       ,
    "bogie"      , "goonie"   , "laddie"      , "pixie"      , "techie"    ,
    "bombie"     , "groupie"  , "laramie"     , "quickie"    , "^tie"      ,
    "collie"     , "hankie"   , "lingerie"    , "reverie"    , "toughie"   ,
    "cookie"     , "hippie"   , "meanie"      , "rookie"     , "valkyrie"  ,
))
singular_irregular = {
       "atlantes": "atlas",
        "atlases": "atlas",
           "axes": "axe",
         "beeves": "beef",
       "brethren": "brother",
       "children": "child",
        "corpora": "corpus",
       "corpuses": "corpus",
    "ephemerides": "ephemeris",
           "feet": "foot",
        "ganglia": "ganglion",
          "geese": "goose",
         "genera": "genus",
          "genii": "genie",
       "graffiti": "graffito",
         "helves": "helve",
           "kine": "cow",
         "leaves": "leaf",
         "loaves": "loaf",
            "men": "man",
      "mongooses": "mongoose",
         "monies": "money",
          "moves": "move",
         "mythoi": "mythos",
         "numena": "numen",
       "occipita": "occiput",
      "octopodes": "octopus",
          "opera": "opus",
         "opuses": "opus",
            "our": "my",
           "oxen": "ox",
          "penes": "penis",
        "penises": "penis",
         "people": "person",
          "sexes": "sex",
    "soliloquies": "soliloquy",
          "teeth": "tooth",
         "testes": "testis",
        "trilbys": "trilby",
         "turves": "turf",
            "zoa": "zoon",
}


def singularize(word, pos=NOUN, custom={}):
    """ Returns the singular of a given word.
    """
    if word in custom:
        return custom[word]
    # Recurse compound words (e.g. mothers-in-law).
    if "-" in word:
        w = word.split("-")
        if len(w) > 1 and w[1] in plural_prepositions:
            return singularize(w[0], pos, custom) + "-" + "-".join(w[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1]) + "'s"
    w = word.lower()
    for x in singular_uninflected:
        if x.endswith(w):
            return word
    for x in singular_uncountable:
        if x.endswith(w):
            return word
    for x in singular_ie:
        if w.endswith(x + "s"):
            return w
    for x in singular_irregular:
        if w.endswith(x):
            return re.sub('(?i)' + x + '$', singular_irregular[x], word)
    for suffix, inflection in singular_rules:
        m = suffix.search(word)
        g = m and m.groups() or []
        if m:
            for k in range(len(g)):
                if g[k] is None:
                    inflection = inflection.replace('\\' + str(k + 1), '')
            return suffix.sub(inflection, word)
    return word

# %%
@dataclass
class ChairConfig:
    INPUT_TYPE: str = "both"

    CANDIDATES_CSV: str = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"
    SELECTIONS_CSV: str = "outputs/coco/coco_llm_rerank_selections.csv"

    INSTANCES_JSON: str = "data/coco/annotations/instances_val2014.json"


chair_config = ChairConfig()

assert chair_config.INPUT_TYPE in ("type1", "type2", "both")

# %%
SYNONYM_LINES = [
    "person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager",
    "bicycle, bike, bicycle, bike, unicycle, minibike, trike",
    "car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi",
    "motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped",
    "airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane",
    "bus, minibus, trolley",
    "train, locomotive, tramway, caboose",
    "truck, pickup, lorry, hauler, firetruck",
    "boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship",
    "traffic light, street light, traffic signal, stop light, streetlight, stoplight",
    "fire hydrant, hydrant",
    "stop sign",
    "parking meter",
    "bench, pew",
    "bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird",
    "cat, kitten, feline, tabby",
    "dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky",
    "horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco",
    "sheep, lamb, ram, lamb, goat, ewe",
    "cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison",
    "elephant",
    "bear, panda",
    "zebra",
    "giraffe",
    "backpack, knapsack",
    "umbrella",
    "handbag, wallet, purse, briefcase",
    "tie, bow, bow tie",
    "suitcase, suit case, luggage",
    "frisbee",
    "skis, ski",
    "snowboard",
    "sports ball, ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard, longboard, skimboard, shortboard, wakeboard",
    "tennis racket, racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife, pocketknife, knive",
    "spoon",
    "bowl, container",
    "banana",
    "apple",
    "sandwich, burger, sub, cheeseburger, hamburger",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut, doughnut, bagel",
    "cake,  cheesecake, cupcake, shortcake, coffeecake, pancake",
    "chair, seat, stool",
    "couch, sofa, recliner, futon, loveseat, settee, chesterfield",
    "potted plant, houseplant",
    "bed",
    "dining table, table, desk",
    "toilet, urinal, commode, toilet, lavatory, potty",
    "tv, monitor, televison, television",
    "laptop, computer, notebook, netbook, lenovo, macbook, laptop computer",
    "mouse",
    "remote",
    "keyboard",
    "cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone",
    "microwave",
    "oven, stovetop, stove, stove top oven",
    "toaster",
    "sink",
    "refrigerator, fridge, fridge, freezer",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear, teddybear",
    "hair drier, hairdryer",
    "toothbrush",
]

COCO_CATEGORIES = [line.split(",")[0].strip() for line in SYNONYM_LINES]
assert len(COCO_CATEGORIES) == 80, f"expected 80 COCO categories, got {len(COCO_CATEGORIES)}"

# %%
class ChairExtractor:
    def __init__(self, synonym_lines: Sequence[str] = SYNONYM_LINES):
        synonyms = [line.strip().split(", ") for line in synonym_lines]
        self.mscoco_objects: List[str] = []
        self.inverse_synonym_dict: Dict[str, str] = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]
        self.mscoco_object_set: Set[str] = set(self.mscoco_objects)

        coco_double_words = [
            "motor bike", "motor cycle", "air plane", "traffic light", "street light",
            "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
            "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
            "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
            "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
            "hot dog", "teddy bear", "home plate", "train track",
        ]
        animal_words = [
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "animal", "cub",
        ]
        vehicle_words = ["jet", "train"]

        self.double_word_dict: Dict[str, str] = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict["baby %s" % animal_word] = animal_word
            self.double_word_dict["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_objects(self, caption: str) -> Tuple[List[str], List[str]]:
        words = nltk.word_tokenize(str(caption).lower())
        words = [singularize(w) for w in words]

        i = 0
        double_words: List[str] = []
        while i < len(words):
            double_word = " ".join(words[i:i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        words = [word for word in words if word in self.mscoco_object_set]
        node_words = [self.inverse_synonym_dict[word] for word in words]
        return words, node_words

    def caption_to_object_set(self, caption: str) -> Set[str]:
        return set(self.caption_to_objects(caption)[1])


extractor = ChairExtractor()

EXTRACTOR_CHECKS = [
    ("a man riding a motorcycle down the street", {"person", "motorcycle"}),
    ("two dogs are playing with a frisbee in the grass", {"dog", "frisbee"}),
    ("a group of people sitting at a dining table with wine glasses", {"person", "dining table", "wine glass"}),
    ("a woman holding a cell phone and a cup of coffee", {"person", "cell phone", "cup"}),
    ("a train traveling down train tracks near a station", {"train"}),
    ("a teddy bear sitting on a bed next to a laptop", {"teddy bear", "bed", "laptop"}),
    ("a baby elephant standing next to an adult elephant", {"elephant"}),
    ("the seat of the toilet is up in the bathroom", {"toilet"}),
    ("a plate of broccoli and carrots next to two sandwiches", {"broccoli", "carrot", "sandwich"}),
    ("a baseball player swinging a bat at home plate", {"person"}),
    ("a couple of hot dogs sitting on a bun", {"hot dog"}),
    ("an orange cat wearing an orange and white sweater", {"cat", "orange"}),
    ("an orange sitting next to a banana and an apple", {"orange", "banana", "apple"}),
    ("a bowl of oranges on a wooden table", {"bowl", "orange", "dining table"}),
    ("a man in an orange shirt watching a tennis match", {"person", "orange"}),
]
for check_caption, expected in EXTRACTOR_CHECKS:
    got = extractor.caption_to_object_set(check_caption)
    assert got == expected, f"extractor check failed: {check_caption!r} -> {got}, expected {expected}"


# %%
def coco_image_id_from_name(image_name: str) -> Optional[int]:
    match = re.search(r"(\d{6,})", str(image_name))
    return int(match.group(1)) if match else None


def load_instance_objects(instances_json: str) -> Dict[int, Set[str]]:
    with open(instances_json, "r") as f:
        data = json.load(f)

    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    imid_to_objects: Dict[int, Set[str]] = {}
    for annotation in data["annotations"]:
        name = id_to_name.get(annotation["category_id"])
        if name is None:
            continue
        node_word = extractor.inverse_synonym_dict[name]
        imid_to_objects.setdefault(annotation["image_id"], set()).add(node_word)

    return imid_to_objects


instance_objects = load_instance_objects(chair_config.INSTANCES_JSON)


# %%
@dataclass
class ImageRecord:
    image_id: int
    image_name: str
    references: List[str]
    gt_objects: Set[str] = field(default_factory=set)
    candidates: Dict[int, str] = field(default_factory=dict)
    selected_caption: Optional[str] = None


def read_references(row: pd.Series, columns: Sequence[str]) -> List[str]:
    refs = []
    for i in range(1, 6):
        col = f"ground_truth_{i}"
        if col in columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                refs.append(str(value))
    return refs


def attach_gt_objects(record: ImageRecord):
    gt_from_captions: Set[str] = set()
    for ref in record.references:
        gt_from_captions |= extractor.caption_to_object_set(ref)

    coco_id = coco_image_id_from_name(record.image_name)
    gt_from_instances = set(instance_objects.get(coco_id, set())) if coco_id is not None else set()

    record.gt_objects = gt_from_captions | gt_from_instances


def load_type1(csv_path: str) -> Dict[int, ImageRecord]:
    df = pd.read_csv(csv_path)

    records: Dict[int, ImageRecord] = {}
    for image_id, group in df.groupby("image_id"):
        group = group.sort_values("candidate_rank")
        first = group.iloc[0]

        record = ImageRecord(
            image_id=int(image_id),
            image_name=str(first["image_name"]),
            references=read_references(first, group.columns),
        )
        for _, row in group.iterrows():
            caption = row["caption"]
            if pd.notna(caption) and str(caption).strip():
                record.candidates[int(row["candidate_rank"])] = str(caption)

        if not record.references or not record.candidates:
            continue
        attach_gt_objects(record)
        records[record.image_id] = record

    if not records:
        raise ValueError(
            f"No usable rows in {csv_path} - every image lacked ground_truth_* columns or captions"
        )

    return records


def load_type2(csv_path: str) -> Dict[int, ImageRecord]:
    df = pd.read_csv(csv_path)

    records: Dict[int, ImageRecord] = {}
    for _, row in df.iterrows():
        caption = row["selected_caption"]
        if pd.isna(caption) or not str(caption).strip():
            continue

        record = ImageRecord(
            image_id=int(row["image_id"]),
            image_name=str(row["image_name"]),
            references=read_references(row, df.columns),
        )
        record.selected_caption = str(caption)

        if not record.references:
            continue
        attach_gt_objects(record)
        records[record.image_id] = record

    if not records:
        raise ValueError(
            f"No usable rows in {csv_path} - every image lacked ground_truth_* columns or a caption"
        )

    return records


# %%
@dataclass
class ChairScore:
    label: str
    chair_i: float
    chair_s: float


def score_captions(pairs: Sequence[Tuple[ImageRecord, str]], label: str) -> ChairScore:
    n_mentions = 0
    n_hallucinated = 0
    n_hallucinated_captions = 0

    for record, caption in pairs:
        _, nodes = extractor.caption_to_objects(caption)
        hallucinated = [obj for obj in nodes if obj not in record.gt_objects]

        n_mentions += len(nodes)
        n_hallucinated += len(hallucinated)
        n_hallucinated_captions += int(len(hallucinated) > 0)

    n_captions = len(pairs)
    return ChairScore(
        label=label,
        chair_i=n_hallucinated / n_mentions if n_mentions else 0.0,
        chair_s=n_hallucinated_captions / n_captions if n_captions else 0.0,
    )


# %%
scores: List[ChairScore] = []

if chair_config.INPUT_TYPE in ("type1", "both"):
    type1_records = load_type1(chair_config.CANDIDATES_CSV)
    pairs = [
        (record, record.candidates[min(record.candidates)])
        for record in type1_records.values()
    ]
    scores.append(score_captions(pairs, "beam top-1"))

if chair_config.INPUT_TYPE in ("type2", "both"):
    type2_records = load_type2(chair_config.SELECTIONS_CSV)
    pairs = [(record, record.selected_caption) for record in type2_records.values()]
    scores.append(score_captions(pairs, "LLM-selected caption"))

print("\nCHAIR  (%, lower is better)")
print(f"  {'caption set':<24}{'CHAIR_s':>9}{'CHAIR_i':>9}")
for score in scores:
    print(f"  {score.label:<24}{score.chair_s * 100:9.2f}{score.chair_i * 100:9.2f}")

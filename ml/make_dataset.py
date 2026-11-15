"""
Generate a labelled NER dataset of government-style project announcements.
Test set uses TEMPLATES THE MODEL NEVER SEES IN TRAINING (tests generalisation, not memorisation).
Real data: label press releases in doccano, then run convert_real.py and mix/replace.
    python make_dataset.py --n 3000
"""
import json, random, re, argparse, os
from common import spans_to_gold

PLACES = "Rohini Dwarka Okhla Najafgarh Shahdara Burari Saket Janakpuri Pitampura Narela Bawana Ghazipur Kalkaji Mayur-Vihar Karol-Bagh Vasant-Kunj Nehru-Place Raipur Bhilai Durg Jhansi Sector-9 Rajouri-Garden Lajpat-Nagar Dilshad-Garden Mundka Alipur Najafgarh".replace("-", " ").split(" ")
PLACES = [p for p in PLACES if p]
KINDS = ["Hospital", "Medical College", "Flyover", "Bridge", "Metro Station", "Stadium", "Park", "Water Treatment Plant",
         "Government School", "Expressway", "Railway Station", "Community Centre", "Trauma Centre", "Bus Terminal"]
SUFFIX = ["", "", "", " Expansion", " Extension", " Upgradation"]
FIRST = "Rajesh Sunita Amit Priya Vikram Anjali Suresh Meena Harish Kavita Rohit Neha Deepak Pooja Manoj Ritu Sanjay Anita".split()
LAST = "Sharma Verma Gupta Singh Yadav Meena Joshi Kapoor Bansal Chauhan Mishra Rao Pandey Tiwari Saxena Malhotra".split()
TITLES = ["Chief Minister", "Union Minister", "Mayor", "Lieutenant Governor", "Commissioner", "Health Minister", "Education Minister", "MLA"]
SCHEMES = ["PMAY", "Smart Cities Mission", "AMRUT", "Ayushman Bharat", "Bharatmala", "Samagra Shiksha", "Jal Jeevan Mission",
           "Swachh Bharat Mission", "Namami Gange", "PM Gati Shakti"]
DEPTS = ["Department of Health & Family Welfare", "Public Works Department", "Ministry of Road Transport and Highways",
         "Delhi Jal Board", "DMRC", "NHAI", "Department of Education", "Municipal Corporation of Delhi", "Delhi Development Authority"]
MONEY_FMT = ["Rs. {} crore", "₹{} Cr", "Rs {} crore", "₹ {} crore", "{} crore rupees", "Rs. {} Cr"]
PCT_FMT = ["{}%", "{} per cent", "{} percent"]
MONTHS = "January February March April May June July August September October November December".split()
DISTR = ["About {a}% of residents surveyed supported the move.", "The facility is expected to serve {a} lakh people annually.",
         "Rs {a} crore was separately allocated for unrelated drainage works in the same budget.",
         "Traffic on the stretch has risen {a} per cent over two years.", "A total of {a} workers were employed on site."]

TEMPLATES = [
 "{TITLE} {LEADER} inaugurated the {PROJECT} today. The project was sanctioned at {BUDGET} and executed by the {DEPT} under {SCHEME}.",
 "The {PROJECT}, estimated to cost {BUDGET}, is {PROGRESS} complete, {TITLE} {LEADER} said on {DATE}. {DISTR}",
 "{DEPT} has spent {SPENT} on the {PROJECT} so far, against an outlay of {BUDGET}. {LEADER}, {TITLE}, reviewed progress.",
 "Speaking at the {PROJECT} site, {TITLE} {LEADER} said {PROGRESS} of the work stands finished. The {BUDGET} project falls under {SCHEME}. {DISTR}",
 "Work worth {SPENT} has been utilised on the {PROJECT}, a {BUDGET} initiative of the {DEPT}. {DISTR}",
 "The {PROJECT} was approved for {BUDGET} under the {SCHEME} programme. {TITLE} {LEADER} laid the foundation stone on {DATE}.",
 "{DISTR} Meanwhile, the {PROJECT} ({BUDGET}) remains {PROGRESS} complete according to the {DEPT}.",
 "{LEADER}, the {TITLE}, dedicated the {PROJECT} to the public. Costing {BUDGET}, it was developed under {SCHEME}.",
 "A review meeting chaired by {TITLE} {LEADER} found that the {PROJECT} has reached {PROGRESS} progress. {SPENT} has been released out of {BUDGET}.",
 "{DEPT} announced that the {PROJECT} will be built at a cost of {BUDGET}. {DISTR} {TITLE} {LEADER} welcomed the decision.",
 "The {PROJECT} is nearing completion, {PROGRESS} done, with {SPENT} spent from the sanctioned {BUDGET}. {DISTR}",
 "On {DATE}, {TITLE} {LEADER} opened the {PROJECT}. {DISTR} The {BUDGET} facility was constructed under {SCHEME} by the {DEPT}.",
 "Under {SCHEME}, the {DEPT} is executing the {PROJECT} at {BUDGET}. {PROGRESS} of the physical work is complete.",
 "{TITLE} {LEADER} criticised delays at the {PROJECT}, where only {PROGRESS} of work is done despite {SPENT} already spent. {DISTR}",
 "{DEPT} has already spent {SPENT} on the {PROJECT}. {DISTR}",
 "So far {SPENT} has been utilised on the {PROJECT}, {TITLE} {LEADER} told reporters on {DATE}.",
 "The {PROJECT} was sanctioned at {BUDGET}, of which {SPENT} has been spent so far. {TITLE} {LEADER} reviewed the {DEPT} report.",
 "Against a sanctioned amount of {BUDGET}, only {SPENT} had been spent on the {PROJECT} by {DATE}, and {PROGRESS} of work was complete.",
 "{LEADER}, {TITLE}, said {BUDGET} had been earmarked for the {PROJECT} while {SPENT} stood utilised.",
 "Spending on the {PROJECT} has touched {SPENT}; the {DEPT} had budgeted {BUDGET}.",
 "The {PROJECT} is {PROGRESS} complete. {DISTR} {SPENT} has been spent.",
 "{TITLE} {LEADER} visited the {PROJECT} on {DATE}. {DISTR} The {DEPT} said work is {PROGRESS} complete.",
 "The {PROJECT} has cost the exchequer {SPENT} to date. {DISTR} Work is {PROGRESS} complete under {SCHEME}.",
 "The {DEPT} approved {BUDGET} for the {PROJECT}; {SPENT} was spent in the first phase, {TITLE} {LEADER} said.",
 "{SPENT} has been spent against the {BUDGET} allotted for the {PROJECT}, under {SCHEME}.",
]
TEST_TEMPLATES = {3, 9, 13, 22, 23, 24}   # never used for train/val
LABEL_KEYS = {"PROJECT", "BUDGET", "SPENT", "PROGRESS", "LEADER", "SCHEME", "DEPT"}


def num():
    n = random.choice([random.randint(5, 900), random.randint(1000, 9000), round(random.uniform(5, 500), 1)])
    return n, (f"{n:,}" if isinstance(n, int) else str(n))


def make_doc(tpl):
    keys = set(re.findall(r"\{(\w+)\}", tpl))
    ent, vals = {}, {}
    ent["PROJECT"] = f"{random.choice(PLACES)} {random.choice(KINDS)}{random.choice(SUFFIX)}"
    for k in ("BUDGET", "SPENT"):
        n, s = num(); vals[k] = n * 1e7; ent[k] = random.choice(MONEY_FMT).format(s)
    p = random.randint(5, 99); vals["PROGRESS"] = float(p); ent["PROGRESS"] = random.choice(PCT_FMT).format(p)
    ent["LEADER"] = f"{random.choice(FIRST)} {random.choice(LAST)}"
    ent["TITLE"] = random.choice(TITLES); ent["SCHEME"] = random.choice(SCHEMES); ent["DEPT"] = random.choice(DEPTS)
    ent["DATE"] = f"{random.randint(1, 28)} {random.choice(MONTHS)} {random.randint(2022, 2026)}"
    ent["DISTR"] = random.choice(DISTR).format(a=random.randint(10, 95))
    out, spans, i = "", [], 0
    for m in re.finditer(r"\{(\w+)\}", tpl):
        out += tpl[i:m.start()]; v = ent[m.group(1)]
        if m.group(1) in LABEL_KEYS:
            spans.append({"start": len(out), "end": len(out) + len(v), "label": m.group(1)})
        out += v; i = m.end()
    out += tpl[i:]
    gold = spans_to_gold(out, spans)
    for k, f in (("BUDGET", "budgetSanctioned"), ("SPENT", "budgetSpent"), ("PROGRESS", "completionPercentage")):
        if k in keys: assert abs(gold[f] - vals[k]) < 1, (out, k, gold[f], vals[k])   # label pipeline sanity check
    return {"text": out, "spans": spans, "gold": gold}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data"); a = ap.parse_args()
    random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    train_ids = [i for i in range(len(TEMPLATES)) if i not in TEST_TEMPLATES]
    docs = [make_doc(TEMPLATES[random.choice(train_ids)]) for _ in range(a.n)]
    test = [make_doc(TEMPLATES[random.choice(sorted(TEST_TEMPLATES))]) for _ in range(max(300, a.n // 5))]
    cut = int(len(docs) * 0.9)
    for name, rows in (("train", docs[:cut]), ("val", docs[cut:]), ("test", test)):
        with open(f"{a.out}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} docs")
    print("sample:", docs[0]["text"])


if __name__ == "__main__":
    main()

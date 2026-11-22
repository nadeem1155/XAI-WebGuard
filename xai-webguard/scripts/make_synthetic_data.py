"""Generate a small synthetic corpus so the pipeline can be smoke-tested without
the real datasets. NOT for reporting results — replace with the real data.

    python scripts/make_synthetic_data.py --out data/synthetic.csv --n 1500
"""
import argparse
import csv
import os
import random

BENIGN = [
    "search=blue running shoes size {n}", "page={n}", "user=alice&sort=name",
    "q=how to bake bread", "id={n}", "city=Lucknow&limit={n}",
    "comment=Great product, loved it!", "name=O'Brien", "lang=en&page={n}",
    "filter=price_asc", "email=user{n}@example.com", "title=The Hungry Caterpillar",
]
SQLI = [
    "id={n} OR 1=1--", "user=admin'--", "id={n} UNION SELECT username,password FROM users",
    "name=' OR 'a'='a", "q={n}; DROP TABLE users;--", "id={n} AND 1=2 UNION SELECT NULL,version()--",
    "login=' OR 1=1 #", "id={n}' AND SLEEP(5)--", "cat={n} UNION ALL SELECT table_name FROM information_schema.tables",
]
XSS = [
    "q=<script>alert(1)</script>", "name=<img src=x onerror=alert(1)>",
    "comment=<svg onload=alert(document.cookie)>", "search=<iframe src=javascript:alert(1)>",
    "redirect=javascript:alert(1)", "msg=<body onload=alert('xss')>",
    "title=<a href=\"javascript:eval('x')\">click</a>", "x=<img src=1 onerror=eval(atob('YWxlcnQoMSk='))>",
]


def fill(t):
    return t.replace("{n}", str(random.randint(1, 9999)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.csv")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    random.seed(0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for _ in range(args.n):
        r = random.random()
        if r < 0.5:
            rows.append((fill(random.choice(BENIGN)), 0))
        elif r < 0.75:
            rows.append((fill(random.choice(SQLI)), 1))
        else:
            rows.append((fill(random.choice(XSS)), 2))
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Query", "Label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

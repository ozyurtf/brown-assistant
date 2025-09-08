import argparse
import json
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import map_code_to_dept_cab

API_URL = "https://cab.brown.edu/api/"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://cab.brown.edu",
    "Referer": "https://cab.brown.edu/",
}

session = requests.Session()
session.headers.update(HEADERS)

def normalize_course_code(code: str) -> str:
    if not code:
        return ""
    m = re.match(r"^\s*([A-Z&]+)\s*-?\s*([0-9]{3,4})", code)
    if m:
        return f"{m.group(1)} {m.group(2).zfill(4)}"
    parts = code.split()
    if len(parts) >= 2:
        m2 = re.match(r"(\d+)", parts[1])
        if m2:
            return f"{parts[0]} {m2.group(1).zfill(4)}"
    return code

def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def search_term(term: str, dept: str | None) -> list[dict]:
    print(f"[search_term] start term={term} dept={dept}", flush=True)
    criteria = [{"field": "is_ind_study", "value": "N"}, {"field": "is_canc", "value": "N"}]
    if dept:
        criteria.append({"field": "dept", "value": dept})
    payload = {"other": {"srcdb": term}, "criteria": criteria}
    response = session.post(API_URL, params={"page": "fose", "route": "search"}, json=payload, timeout=30)
    response.raise_for_status()
    results = response.json().get("results", [])
    print(f"[search_term] done term={term} results={len(results)}", flush=True)
    return results


def fetch_details(term: str, course_code: str, crn: str) -> dict:
    print(f"[fetch_details] start term={term} code={course_code} crn={crn}", flush=True)
    payload = {
        "group": f"code:{course_code}",
        "key": f"crn:{crn}",
        "srcdb": term,
        "matched": f"crn:{crn}",
        "userWithRolesStr": "!!!!!!",
    }
    response = session.post(API_URL, params={"page": "fose", "route": "details"}, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    print(f"[fetch_details] done term={term} code={course_code}", flush=True)
    return data


def build_output(results: list[dict], workers: int) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for r in results:
        term = str(r.get("srcdb") or "").strip()
        if not term:
            continue
        code = normalize_course_code(r.get("code") or "")
        if not code:
            continue
        grouped.setdefault(term, {}).setdefault(code, []).append(r)
    
    terms_map: dict = {}
    for term, by_course in grouped.items():
        print(f"[build_output] term={term} courses={len(by_course)}", flush=True)
        term_bucket = terms_map.setdefault(term, {"dept": {}})
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(by_course)))) as pool:
            future_to_meta = {}
            for code, sections in by_course.items():
                primary = sections[0]
                crn = str(primary.get("crn", "")).strip()
                fut = pool.submit(fetch_details, term, code.replace(" ", "_"), crn)
                future_to_meta[fut] = (code, sections)
            print(f"[build_output] submitted detail futures={len(future_to_meta)} for term={term}", flush=True)
            for fut in as_completed(future_to_meta):
                code, sections = future_to_meta[fut]
                try:
                    details = fut.result()
                except Exception as e:
                    print(f"[build_output] details failed code={code} err={e}", flush=True)
                    details = {}
                print(f"[build_output] details done code={code}", flush=True)
                dept = code.split()[0].upper()
                bucket = term_bucket["dept"].setdefault(dept, [])
                p = sections[0]
                instr_html = details.get("instructordetail_html", "")
                email_m = re.search(r"mailto:([^'\"]+)", instr_html or "")
                meeting = clean_html(details.get("meeting_html", "")) or p.get("meets", "")
                sections_text = "\n".join(
                    f"- CRN {sect.get('crn','')}: {sect.get('no','')} | {sect.get('instr','')} | {sect.get('meets','')} | {sect.get('stat','')}"
                    for sect in sections
                )
                course_obj = {
                    "course_id": code.replace(" ", "_"),
                    "course": code,
                    "title": p.get("title", ""),
                    "total_sections": len(sections),
                    "instructor_name": p.get("instr", ""),
                    "instructor_email": email_m.group(1) if email_m else "",
                    "meeting_times": meeting,
                    "description": clean_html(details.get("description", "")),
                    "registration_restrictions": clean_html(details.get("registration_restrictions", "")),
                    "course_attributes": clean_html(details.get("attr_html", "")),
                    "exam_info": clean_html(details.get("exam_html", "")),
                    "class_notes": clean_html(details.get("clssnotes", "")),
                    "sections_text": sections_text,
                }
                bucket.append(course_obj)

    return {term: term_data.get("dept", {}) for term, term_data in terms_map.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", nargs="+", default=["202510"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="cab.json")
    args = parser.parse_args()

    all_results: list[dict] = []
    start = time.time()
    targets: list[tuple[str, str | None]] = []
    
    cab_code_to_dept_map = map_code_to_dept_cab()
    dept_codes = list(cab_code_to_dept_map.values())
    targets = [(t, d) for t in args.terms for d in dept_codes]
    print(f"[main] discovered departments={len(dept_codes)} total_targets={len(targets)}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(search_term, term, dept) for term, dept in targets]
        for f in as_completed(futs):
            try:
                res = f.result()
                all_results.extend(res)
                print(f"[main] collected batch results={len(res)} total={len(all_results)}", flush=True)
            except Exception as e:
                print(f"[main] search failed: {e}", flush=True)
    print(f"[main] searches done in {time.time()-start:.1f}s total_results={len(all_results)}", flush=True)

    data = build_output(all_results, args.workers)
    with open(args.out, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[main] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
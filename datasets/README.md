# India demonstration dataset

`india_demo_100.json` contains **100 fictional job listings and 100 fictional
resumes**, following the same `resumes` / `jds` JSON structure as the original
five examples. It is synthetic demonstration data, not scraped vacancies or
verified candidate records, and should not be treated as an accuracy benchmark.

It covers 20 technology role families, five profile variants per family, and
20 Indian cities. Resumes state skills, experience, projects and education;
jobs distinguish essential requirements, minimum experience and preferred skills.
The junior/associate profiles omit some essential skills, experienced/senior
profiles evidence more requirements, and transition profiles expose clearer gaps.
OR requirements are included to demonstrate alternatives rather than requiring both.

`demo_pairs` links profiles to corresponding role-family examples for exploration;
these are illustrative pairs, not numerical ground-truth labels. Location and
work arrangement are independent ranking preferences. Mumbai, Navi Mumbai, Pune,
Ratnagiri and Ahmedabad are included for the +5/+4/+3/+2/0 location demonstration.

Use **Jobs → Demo dataset → India dataset (100 jobs + 100 resumes) → Load dataset**.
The loader adds these records and preserves existing jobs/resumes. Repeating it
reuses matching stored jobs and skips duplicate resumes. Resumes persist in the
browser; jobs persist in the backend. When the complete demo job dataset is
already saved, opening the app in another browser imports its demo resumes once.
Your selected resume and uploaded resumes are preserved. Removing a demo resume
is respected on later refreshes; use Load dataset to restore it deliberately.
Uploading real resumes still works normally.

Reproduce the JSON with `.venv/bin/python scripts/generate_india_demo.py`.

"""Reproduce the synthetic India demo using the original sample JSON format."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["Mumbai", "Navi Mumbai", "Pune", "Ratnagiri", "Ahmedabad", "Bengaluru",
          "Hyderabad", "Chennai", "Delhi", "Kolkata", "Noida", "Gurugram", "Jaipur",
          "Kochi", "Indore", "Lucknow", "Nagpur", "Thane", "Nashik", "Coimbatore"]
# role key, resume label, job title, essential skills, preferred skills, demonstrated work
ROLES = [
    ("python_backend", "Python Developer", "Python Backend Engineer", "Python|FastAPI|PostgreSQL|Docker|REST APIs|microservices", "Redis|AWS|Pytest|GitHub Actions", "building tested order-management APIs and database-backed services"),
    ("react_frontend", "Frontend Engineer", "React Frontend Developer", "React|TypeScript|JavaScript|HTML5|CSS3|Redux", "Next.js|Tailwind CSS|Jest|Cypress", "building accessible customer dashboards with reusable components and state management"),
    ("data_science", "Data Scientist", "Data Scientist", "Python|Pandas|NumPy|Scikit-learn|SQL|statistics", "PyTorch|Matplotlib|Streamlit|model evaluation", "developing demand-forecasting models with reproducible validation and error analysis"),
    ("devops", "DevOps Engineer", "DevOps and Cloud Engineer", "Docker|Kubernetes|Terraform|Linux|Bash|CI/CD", "AWS|Ansible|Jenkins|GitLab CI/CD", "automating container deployments, infrastructure provisioning and operational monitoring"),
    ("fullstack", "Fullstack Engineer", "Full Stack Software Engineer", "Python or Node.js|React|PostgreSQL|REST APIs|Git|JavaScript", "MongoDB|Nginx|Google Cloud Platform|Docker", "delivering inventory-management applications with frontend screens and backend services"),
    ("java_backend", "Java Developer", "Java Backend Engineer", "Java|Spring Boot|Hibernate|PostgreSQL|REST APIs|JUnit", "Kafka|Docker|Redis|Maven", "building payment-reconciliation services with transaction handling and integration tests"),
    ("data_engineer", "Data Engineer", "Data Engineer", "Python|SQL|Apache Spark|Airflow|ETL|data modelling", "dbt|Snowflake|AWS S3|data quality testing", "maintaining scheduled ingestion pipelines and analytics-ready warehouse tables"),
    ("android", "Android Developer", "Android Application Developer", "Kotlin|Android SDK|Jetpack Compose|Room|REST APIs|Git", "Firebase|Gradle|JUnit|Coroutines", "building offline-capable delivery apps with API integration and local persistence"),
    ("ios", "iOS Developer", "iOS Application Developer", "Swift|SwiftUI|Core Data|REST APIs|Git|XCTest", "Combine|UIKit|Fastlane|accessibility testing", "developing appointment-booking apps with secure session handling and automated tests"),
    ("qa", "QA Automation Engineer", "QA Automation Engineer", "Python|Selenium|Pytest|API testing|Git|test planning", "Playwright|Postman|JMeter|CI/CD", "maintaining browser and API regression suites with reproducible defect reports"),
    ("security", "Security Analyst", "Cybersecurity Analyst", "Linux|network security|SIEM|incident response|vulnerability assessment|Python", "Splunk|Wireshark|OWASP|security documentation", "triaging security alerts and documenting investigation and remediation workflows"),
    ("cloud", "Cloud Engineer", "Cloud Platform Engineer", "AWS|Terraform|Linux|IAM|Docker|Python", "Azure|Ansible|Kubernetes|CloudWatch", "provisioning cloud services with least-privilege access and recovery procedures"),
    ("ml", "Machine Learning Engineer", "Machine Learning Engineer", "Python|PyTorch or TensorFlow|Scikit-learn|Docker|MLflow|FastAPI", "LangChain|model fine-tuning|ChromaDB|monitoring", "deploying versioned model-serving APIs with evaluation and inference monitoring"),
    ("dotnet", ".NET Developer", ".NET Backend Developer", "C#|ASP.NET Core|Entity Framework Core|SQL Server|REST APIs|Git", "Azure|Redis|Docker|xUnit", "building business workflow APIs with relational data access and authentication"),
    ("ux", "UX Designer", "UI and UX Designer", "Figma|wireframing|prototyping|user research|usability testing|design systems", "HTML5|CSS3|Adobe XD|accessibility", "designing and testing onboarding journeys with documented usability findings"),
    ("bi", "Business Intelligence Analyst", "BI Analyst", "SQL|Excel|Power BI|DAX|data modelling|dashboard design", "Python|Tableau|Snowflake|statistics", "building sales-performance dashboards with governed metrics and drill-down reports"),
    ("dba", "Database Administrator", "PostgreSQL Database Administrator", "PostgreSQL|SQL|backup and recovery|replication|performance tuning|Linux", "MySQL|Redis|AWS RDS|monitoring", "maintaining database backups, replication and query-performance runbooks"),
    ("embedded", "Embedded Software Engineer", "Embedded Software Engineer", "C|C++|microcontrollers|UART|SPI|RTOS", "I2C|Linux|unit testing|debugging", "developing sensor-device firmware with peripheral integration and hardware diagnostics"),
    ("product_analytics", "Product Analyst", "Product Data Analyst", "SQL|Python|Excel|A/B testing|product analytics|statistics", "Amplitude|Google Analytics|Tableau|cohort analysis", "analysing conversion funnels and experiments with documented metric definitions"),
    ("support", "Technical Support Engineer", "Technical Support Engineer", "Linux|SQL|networking|troubleshooting|ticket management|REST APIs", "Python|AWS|ITIL|technical writing", "investigating customer incidents and documenting reproducible fixes and escalation steps"),
]
FIRST = ["Aarav", "Aditi", "Rohan", "Ishita", "Kabir", "Meera", "Arjun", "Ananya", "Vikram", "Nisha"]
LAST = ["Sharma", "Patil", "Iyer", "Shah", "Rao", "Deshmukh", "Menon", "Gupta", "Bose", "Kulkarni"]
LEVELS = [("Junior", 0, 1), ("Associate", 2, 3), ("", 3, 4), ("Senior", 5, 7), ("", 4, 2)]
DEGREES = ["Bachelor of Computer Applications", "B.Tech in Computer Science",
           "B.Sc in Computer Science", "Master of Computer Applications", "B.Tech in Information Technology"]
ARRANGEMENTS = ["On-site", "Hybrid", "Remote", "Hybrid", "On-site"]


def build_dataset():
    resumes, jobs, pairs = [], [], []
    for role_index, (key, label, title, essential, preferred, work) in enumerate(ROLES):
        essential_skills, preferred_skills = essential.split("|"), preferred.split("|")
        for variant, (prefix, minimum_years, experience_years) in enumerate(LEVELS):
            ordinal = role_index * 5 + variant + 1
            city = CITIES[(role_index * 5 + variant) % len(CITIES)]
            home = city
            arrangement = ARRANGEMENTS[variant]
            person = f"{FIRST[(ordinal - 1) % len(FIRST)]} {LAST[(ordinal - 1) // len(FIRST)]}"
            job_id, resume_id = f"jd_india_{key}_{variant + 1:02}", f"res_india_{key}_{variant + 1:02}"
            # Specific alternatives are evidenced by one branch, rather than
            # incorrectly requiring both technologies in an OR requirement.
            core = [skill.split(" or ")[0] for skill in essential_skills]
            if variant == 0:
                evidenced, extras = core[:4], preferred_skills[:1]
            elif variant == 1:
                evidenced, extras = core[:-1], preferred_skills[:2]
            elif variant in (2, 3):
                evidenced, extras = core, preferred_skills[:variant + 1]
            else:
                evidenced, extras = core[:2] + ["Git", "Excel", "technical documentation"], preferred_skills[:1]
            candidate_work = work if variant != 4 else f"completing supervised practice projects in {core[0]} while transitioning from application support"
            candidate = (f"{label} based in {home}, India with {experience_years} years of professional experience. "
                         f"Skilled in {', '.join(evidenced)}. Additional experience with {', '.join(extras)}. "
                         f"Hands-on work {candidate_work}. Collaborated with engineering and operations teams, "
                         f"used Git for version control, and documented project decisions. Education: {DEGREES[variant]}. "
                         f"Interested in {arrangement.lower()} roles and further developing role-specific skills.")
            job_title = f"{prefix} {title}".strip()
            qualification = (f"At least {minimum_years} years of relevant professional experience required."
                             if minimum_years else "Fresh graduates or candidates with up to 2 years of experience may apply.")
            job_text = (f"We are looking for a {job_title} for our {city}, India team. "
                        f"Essential Requirements: {', '.join(essential_skills)}. {qualification} "
                        f"Responsibilities include {work}, collaborating with teammates, and documenting reliable delivery. "
                        f"Bachelor's degree in Computer Science, Information Technology, Computer Applications, "
                        f"or equivalent relevant experience required. Preferred: {', '.join(preferred_skills)}. "
                        f"Work arrangement: {arrangement}. This is a fictional vacancy for project demonstration.")
            resumes.append({"id": resume_id, "name": f"{person} — {label} ({home})", "text": candidate,
                            "location": f"{home}, India", "synthetic": True})
            jobs.append({"id": job_id, "title": job_title, "company": f"Demo {LAST[role_index % 10]} Digital Labs",
                         "location": f"{city}, India ({arrangement})", "text": job_text, "synthetic": True})
            pairs.append({"resume_id": resume_id, "jd_id": job_id, "role_family": key,
                          "profile": ["junior_partial", "associate_partial", "experienced", "senior", "career_transition"][variant]})
    return {"description": "Synthetic India demo: 100 jobs and 100 resumes, shaped like the original five samples. Not real vacancies, people, or ground-truth fit scores.",
            "resumes": resumes, "jds": jobs, "demo_pairs": pairs}


if __name__ == "__main__":
    target = ROOT / "datasets" / "india_demo_100.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(build_dataset(), indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {target}: 100 jobs and 100 resumes")

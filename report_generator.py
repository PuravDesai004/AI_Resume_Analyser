import datetime
from typing import Any


def generate_markdown_report(analysis: dict[str, Any], resume_name: str, jd_title: str) -> str:
    """
    Generates a well-formatted Markdown string from a DetailAnalysis dict.

    Includes:
    - Header: # Gap Analysis Report: {resume_name} vs {jd_title}
    - Date generated
    - Match score (as a heading)
    - Summary section
    - Strengths (bulleted list)
    - Weaknesses (bulleted list)
    - Skill Gap table: Matched, Missing Essential, Missing Preferred, Extra
    - Requirement Match table: Category | Match %
    - Recommendations table: Skill | Reason
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_score = analysis.get("match_score", 0)
    summary = analysis.get("summary", "No summary provided.")
    strengths = analysis.get("strengths", [])
    weaknesses = analysis.get("weaknesses", [])
    skill_gap = analysis.get("skill_gap", {})
    req_match = analysis.get("requirement_match", [])
    recommendations = analysis.get("recommendations", [])

    lines: list[str] = [
        f"# Gap Analysis Report: {resume_name} vs {jd_title}",
        "",
        f"**Date Generated:** {date_str}",
        f"**Candidate Resume:** {resume_name} (ID: {analysis.get('resume_id', 'N/A')})",
        f"**Target Role:** {jd_title} (ID: {analysis.get('jd_id', 'N/A')})",
        "",
        f"## Match Score: {match_score}%",
        "",
        "## Executive Fit Summary",
        summary,
        "",
        "## Key Strengths",
    ]

    if strengths:
        for s in strengths:
            lines.append(f"- {s}")
    else:
        lines.append("- No specific strengths identified.")
    lines.append("")

    lines.append("## Qualification Gaps and Weaknesses")
    if weaknesses:
        for w in weaknesses:
            lines.append(f"- {w}")
    else:
        lines.append("- No specific weaknesses identified.")
    lines.append("")

    # Skill Gap Table
    lines.append("## Skill Gap Analysis")
    lines.append("| Category | Skills |")
    lines.append("|---|---|")
    matched_skills = [
        m.get("skill_name", str(m)) if isinstance(m, dict) else str(m)
        for m in skill_gap.get("matched", [])
    ]
    lines.append(f"| **Matched Skills** | {', '.join(matched_skills) if matched_skills else 'None'} |")
    missing_ess = skill_gap.get("missing_essential", [])
    lines.append(f"| **Missing Essential** | {', '.join(missing_ess) if missing_ess else 'None'} |")
    missing_pref = skill_gap.get("missing_preferred", [])
    lines.append(f"| **Missing Preferred** | {', '.join(missing_pref) if missing_pref else 'None'} |")
    extra_skills = skill_gap.get("extra", [])
    lines.append(f"| **Candidate Extra** | {', '.join(extra_skills) if extra_skills else 'None'} |")
    lines.append("")

    # Requirement Match Table
    lines.append("## Requirement Match")
    lines.append("| Category | Match % |")
    lines.append("|---|---|")
    if req_match:
        for r in req_match:
            cat = r.get("category", "")
            pct = r.get("match_percent", 0)
            lines.append(f"| {cat} | {pct}% |")
    else:
        lines.append("| N/A | N/A |")
    lines.append("")

    # Recommendations Table
    lines.append("## Targeted Recommendations")
    lines.append("| Skill | Reason |")
    lines.append("|---|---|")
    if recommendations:
        for rec in recommendations:
            sk = str(rec.get("skill", "")).replace("|", "/")
            rs = str(rec.get("reason", "")).replace("|", "/")
            lines.append(f"| {sk} | {rs} |")
    else:
        lines.append("| None | No specific recommendations needed. |")
    lines.append("")

    return "\n".join(lines)

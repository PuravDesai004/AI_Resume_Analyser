import datetime
from typing import Any


def _clean_table_cell(text: Any) -> str:
    """
    Sanitizes arbitrary text for safe inclusion within a Markdown table cell.
    - Replaces pipe characters with escaped pipes (\\|) so table columns don't break.
    - Replaces newlines with spaces so table rows remain contiguous on a single line.
    - Strips leading and trailing whitespace.
    """
    if text is None:
        return ""
    s = str(text)
    s = s.replace("|", r"\|")
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return s.strip()


def generate_markdown_report(analysis: dict[str, Any], resume_name: str, jd_title: str) -> str:
    """
    Generates a well-formatted Markdown string from a DetailAnalysis dict.

    Includes:
    - Header: # Gap Analysis Report: {resume_name} vs {jd_title}
    - Date generated
    - Match score (as a heading)
    - Context sufficiency status (with explanation if insufficient)
    - Summary section
    - Strengths (bulleted list)
    - Weaknesses (bulleted list)
    - Skill Gap table: Matched, Missing Essential, Missing Preferred, Extra
    - Requirement Match table: Category | Match %
    - Recommendations table: Skill | Reason
    """
    if not isinstance(analysis, dict):
        analysis = {}

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_score = analysis.get("match_score", 0)
    context_sufficient = analysis.get("context_sufficient", True)
    insufficient_reason = analysis.get("insufficient_reason")
    summary = analysis.get("summary") or "No summary provided."
    strengths = analysis.get("strengths") or []
    weaknesses = analysis.get("weaknesses") or []
    skill_gap = analysis.get("skill_gap") or {}
    req_match = analysis.get("requirement_match") or []
    recommendations = analysis.get("recommendations") or []

    lines: list[str] = [
        f"# Gap Analysis Report: {_clean_table_cell(resume_name)} vs {_clean_table_cell(jd_title)}",
        "",
        f"**Date Generated:** {date_str}",
        f"**Candidate Resume:** {resume_name} (ID: {analysis.get('resume_id', 'N/A')})",
        f"**Target Role:** {jd_title} (ID: {analysis.get('jd_id', 'N/A')})",
        f"**Context Sufficient:** {'Yes' if context_sufficient else 'No'}",
        "",
        f"## Match Score: {match_score}%",
        "",
    ]

    # Handle insufficient context banner
    if not context_sufficient:
        lines.extend([
            "> [!WARNING]",
            f"> **Context Insufficient Gate Triggered:** {insufficient_reason or 'Input document was too short or lacked required detail for reliable skill grounding.'}",
            "> Automated LLM generation was safely bypassed.",
            "",
        ])

    lines.extend([
        "## Executive Fit Summary",
        summary,
        "",
        "## Key Strengths",
    ])

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

    # Skill Gap Table (Sanitized)
    lines.append("## Skill Gap Analysis")
    lines.append("| Category | Skills |")
    lines.append("|---|---|")
    matched_skills = [
        _clean_table_cell(m.get("skill_name", str(m)) if isinstance(m, dict) else str(m))
        for m in skill_gap.get("matched", [])
    ]
    lines.append(f"| **Matched Skills** | {', '.join(matched_skills) if matched_skills else 'None'} |")
    missing_ess = [_clean_table_cell(s) for s in skill_gap.get("missing_essential", [])]
    lines.append(f"| **Missing Essential** | {', '.join(missing_ess) if missing_ess else 'None'} |")
    missing_pref = [_clean_table_cell(s) for s in skill_gap.get("missing_preferred", [])]
    lines.append(f"| **Missing Preferred** | {', '.join(missing_pref) if missing_pref else 'None'} |")
    extra_skills = [_clean_table_cell(s) for s in skill_gap.get("extra", [])]
    lines.append(f"| **Candidate Extra** | {', '.join(extra_skills) if extra_skills else 'None'} |")
    lines.append("")

    # Requirement Match Table (Sanitized)
    lines.append("## Requirement Match")
    lines.append("| Category | Match % |")
    lines.append("|---|---|")
    if req_match:
        for r in req_match:
            if isinstance(r, dict):
                cat = _clean_table_cell(r.get("category", ""))
                pct = r.get("match_percent", 0)
                lines.append(f"| {cat} | {pct}% |")
    else:
        lines.append("| N/A | N/A |")
    lines.append("")

    # Recommendations Table (Sanitized)
    lines.append("## Targeted Recommendations")
    lines.append("| Skill | Reason |")
    lines.append("|---|---|")
    if recommendations:
        for rec in recommendations:
            if isinstance(rec, dict):
                sk = _clean_table_cell(rec.get("skill", ""))
                rs = _clean_table_cell(rec.get("reason", ""))
                lines.append(f"| {sk} | {rs} |")
    else:
        lines.append("| None | No specific recommendations needed. |")
    lines.append("")

    return "\n".join(lines)

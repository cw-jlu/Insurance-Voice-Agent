"""Rebuild the human-reviewed summary for the frozen exp44 text cohort.

This script performs no model calls. It combines the saved responses with the
manual case decisions below and checks citation structure separately from
semantic support.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "exp44_task_release"
PLAN = json.loads((ROOT / "results" / "exp35_plan.json").read_text(encoding="utf-8"))
REQUIRED_HANDOFF = {
    item["id"] for item in PLAN["policies_not_sent_to_model"]
    if item.get("required_external")
}

# Human review of the final spoken text against the original question and the
# corresponding frozen evidence. Explanations retain the concrete reason.
FAILURES = {
    "C12": "Assumes the base-plan annual deductible remains unused when that fact was not provided.",
    "C18": "Applies only the first-enrollment age cap and omits the specified-window re-enrollment rule for age 56+.",
    "C19": "Gives only part of the genetic-testing/drug pathway and misses a required handoff for unresolved material.",
    "C28": "Adds that depression is outside the 100 critical illnesses without the enumerated definitions in this turn's material.",
    "Y033": "Quotes age-69 standard premium without qualifying the first-enrollment versus specified-window re-enrollment rate scope.",
    "Y034": "Quotes age-69 standard premium without qualifying the first-enrollment versus specified-window re-enrollment rate scope.",
}

def main():
    cases = {case["id"]: case for case in PLAN["cases"]}
    decisions = {}
    counts = {"answers": 0, "clarifications": 0, "handoffs": 0,
              "api_errors": 0, "citation_valid": 0, "citation_missing": 0}
    for case_id, case in cases.items():
        path = RESULTS / "text" / f"{case_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        turn = record.get("turn", {})
        timings = turn.get("timings", {})
        consultation = timings.get("consultation", {})
        text = turn.get("text", "")
        status = consultation.get("status")
        error = next((call.get("error") or call.get("error_type")
                      for call in record.get("api_calls", [])
                      if call.get("error") or call.get("error_type")), None)
        if not text and error:
            disposition = "service_error"
            counts["api_errors"] += 1
            handling_ok = False
            citation_ok = None
        else:
            detail = timings.get("answer_detail", {})
            evidence_result = detail.get("evidence_result", {})
            provenance = evidence_result.get("provenance", {})
            citation_ok = provenance.get("valid")
            if citation_ok:
                counts["citation_valid"] += 1
            elif text and case_id != "Y041":
                counts["citation_missing"] += 1
            if status == "handed_off":
                disposition = "safe_handoff"
                counts["handoffs"] += 1
                handling_ok = True
            elif status == "pending":
                disposition = "clarify"
                counts["clarifications"] += 1
                handling_ok = True
            elif case_id == "Y041":
                disposition = "identity_template"
                handling_ok = True
            else:
                disposition = "answer"
                counts["answers"] += 1
                handling_ok = case_id not in FAILURES
        if case_id in FAILURES:
            handling_ok = False
        handoff_request = (timings.get("answer_detail", {})
                           .get("evidence_result", {}).get("handoff_request"))
        if status == "handed_off" and not handoff_request:
            handling_ok = False
        decisions[case_id] = {
            "disposition": disposition,
            "handling_correct": handling_ok,
            "citation_structure_valid": citation_ok,
            "handoff_request_saved": bool(handoff_request),
            "reason": FAILURES.get(case_id, ""),
        }

    required_handoff_met = sorted(
        case_id for case_id in REQUIRED_HANDOFF
        if decisions[case_id]["disposition"] == "safe_handoff"
        and decisions[case_id]["handoff_request_saved"]
    )
    passed = sum(item["handling_correct"] for item in decisions.values())
    semantically_supported = sum(
        item["handling_correct"] and item["citation_structure_valid"] is True
        for item in decisions.values()
    )
    report = {
        "date": "2026-10-14",
        "method": "One-person manual review of all 143 generated responses, with frozen evidence and questions; citation structure is separately read from the saved automatic provenance validator. No second reviewer or blind holdout.",
        "input": "results/exp44_task_release/text",
        "counts": counts,
        "overall_correct_handling": {"correct": passed, "n": len(decisions), "percent": round(passed / len(decisions) * 100, 1)},
        "automatic_answer_accuracy": {"correct": counts["answers"] - len([x for x in FAILURES if decisions[x]["disposition"] == "answer"]), "n": counts["answers"], "percent": round((counts["answers"] - len([x for x in FAILURES if decisions[x]["disposition"] == "answer"])) / counts["answers"] * 100, 1)},
        "semantic_source_support": {"correct": semantically_supported, "n": counts["citation_valid"], "percent": round(semantically_supported / counts["citation_valid"] * 100, 1)},
        "required_handoff": {"correct": len(required_handoff_met), "n": len(REQUIRED_HANDOFF), "missed": sorted(REQUIRED_HANDOFF - set(required_handoff_met))},
        "handoff_requests_saved": {"correct": sum(x["handoff_request_saved"] for x in decisions.values()), "n": counts["handoffs"]},
        "failures": FAILURES,
        "decisions": decisions,
    }
    out = RESULTS / "manual_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["counts", "overall_correct_handling", "automatic_answer_accuracy", "semantic_source_support", "required_handoff", "handoff_requests_saved"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

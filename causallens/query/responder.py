"""
causallens.query.responder
===========================
Generate plain-English answers from counterfactual results.

Takes the output of CounterfactualEngine and turns it into a
human-readable sentence with key statistics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from causallens.query.parser import ParsedQuery, QueryType, InterventionType


class QueryResponder:
    """
    Generate natural language responses from causal inference results.

    Examples
    --------
    >>> responder = QueryResponder()
    >>> answer = responder.respond(parsed_query, counterfactual_result)
    >>> print(answer)
    "If ad_spend were set to 10000, revenue would change from
     $50,234 to $62,891 — an increase of $12,657 (+25.2%)."
    """

    def respond(
        self,
        query: ParsedQuery,
        result: Dict[str, Any],
    ) -> str:
        """
        Generate a plain-English answer.

        Parameters
        ----------
        query : ParsedQuery
            The parsed natural language query.
        result : dict
            Output from CounterfactualEngine.counterfactual() or .ate().

        Returns
        -------
        str : Human-readable answer.
        """
        if query.query_type == QueryType.COUNTERFACTUAL:
            return self._respond_counterfactual(query, result)
        elif query.query_type == QueryType.ATE:
            return self._respond_ate(query, result)
        elif query.query_type == QueryType.CAUSAL_CHECK:
            return self._respond_causal_check(query, result)
        else:
            return "Unable to generate a response for this query type."

    def _respond_counterfactual(
        self,
        query: ParsedQuery,
        result: Dict[str, Any],
    ) -> str:
        """Build answer for counterfactual queries."""
        target = result["target"]
        factual = result["factual_value"]
        cf = result["counterfactual_value"]
        effect = result["effect"]
        rel = result.get("relative_effect", 0)

        # Build the intervention description
        intervention_desc = self._describe_intervention(query)

        # Direction
        if abs(effect) < 1e-10:
            direction = "remain essentially unchanged"
            change_str = ""
        elif effect > 0:
            direction = "increase"
            change_str = f" — an increase of {self._fmt(abs(effect))} (+{abs(rel):.1f}%)"
        else:
            direction = "decrease"
            change_str = f" — a decrease of {self._fmt(abs(effect))} (-{abs(rel):.1f}%)"

        if abs(effect) < 1e-10:
            return (
                f"{intervention_desc}, {target} would "
                f"{direction} at {self._fmt(factual)}."
            )

        return (
            f"{intervention_desc}, {target} would change "
            f"from {self._fmt(factual)} to {self._fmt(cf)}{change_str}."
        )

    def _respond_ate(
        self,
        query: ParsedQuery,
        result: Dict[str, Any],
    ) -> str:
        """Build answer for ATE queries."""
        ate = result["ate"]
        std = result.get("std_effect", 0)
        n = result.get("n_samples", 0)
        var = query.intervention_var
        target = query.target

        if abs(ate) < 1e-10:
            return (
                f"{var} has no significant causal effect on {target} "
                f"(ATE ≈ 0, n={n})."
            )

        direction = "increases" if ate > 0 else "decreases"
        return (
            f"A unit increase in {var} {direction} {target} by "
            f"{self._fmt(abs(ate))} on average "
            f"(ATE = {self._fmt(ate)}, std = {self._fmt(std)}, n={n})."
        )

    def _respond_causal_check(
        self,
        query: ParsedQuery,
        result: Dict[str, Any],
    ) -> str:
        """Build answer for causal check queries."""
        var = query.intervention_var
        target = query.target
        has_path = result.get("has_causal_path", False)
        is_direct = result.get("is_direct_cause", False)

        if is_direct:
            return (
                f"Yes — {var} is a direct cause of {target} "
                f"(there is a directed edge {var} → {target} in the "
                f"causal graph)."
            )
        elif has_path:
            path = result.get("causal_path", [])
            path_str = " → ".join(path) if path else f"{var} → ... → {target}"
            return (
                f"Yes — {var} causes {target} indirectly through "
                f"the path: {path_str}."
            )
        else:
            return (
                f"No — there is no causal path from {var} to {target} "
                f"in the discovered causal graph."
            )

    def _describe_intervention(self, query: ParsedQuery) -> str:
        """Describe the intervention in natural language."""
        var = query.intervention_var
        val = query.intervention_value
        int_type = query.intervention_type

        if int_type == InterventionType.SET:
            return f"If {var} were set to {self._fmt(val)}"
        elif int_type == InterventionType.INCREASE_BY:
            return f"If {var} were increased by {self._fmt(val)}"
        elif int_type == InterventionType.DECREASE_BY:
            return f"If {var} were decreased by {self._fmt(val)}"
        elif int_type == InterventionType.MULTIPLY_BY:
            word = {2.0: "doubled", 3.0: "tripled", 0.5: "halved",
                    4.0: "quadrupled"}.get(val, f"multiplied by {val}")
            return f"If {var} were {word}"
        elif int_type == InterventionType.INCREASE_PCT:
            return f"If {var} increased by {val:.0f}%"
        elif int_type == InterventionType.DECREASE_PCT:
            return f"If {var} decreased by {val:.0f}%"
        else:
            return f"If {var} changed"

    @staticmethod
    def _fmt(value: float) -> str:
        """Smart formatting: integers stay integers, floats get 2 decimals."""
        if value is None:
            return "N/A"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value - round(value)) < 0.01:
            return str(int(round(value)))
        return f"{value:.2f}"

"""
causallens.query.parser
========================
Parse natural language causal queries into structured interventions.

This is the UNIQUE FEATURE of CausalLens — no other open-source causal
AI library (not Salesforce CausalAI, not DoWhy, not CausalNex) offers
natural language counterfactual queries.

How it works
------------
Pure regex + keyword matching (no LLM, no API, no GPU needed).

Supported query patterns
------------------------
1. "What would {target} be if {var} was/were {value}?"
2. "What happens to {target} if we increase {var} by {value}?"
3. "What happens to {target} if we decrease {var} by {value}?"
4. "What happens to {target} if we double/triple/halve {var}?"
5. "If {var} had been {value}, what would {target} have been?"
6. "What is the effect of {var} on {target}?"
7. "Does {var} cause {target}?"
8. "What would {target} be if {var} was set to {value}?"
9. "How would {target} change if {var} increased/decreased by {pct}%?"

Variable matching
-----------------
The parser fuzzy-matches variable names from the query against the
actual column names in the dataset.  It handles:
- Exact matches: "revenue" → "revenue"
- Underscore/space normalization: "ad spend" → "ad_spend"
- Case insensitivity: "Revenue" → "revenue"
- Partial matches: "traffic" → "website_traffic"

Dependencies
------------
- ``causallens.types`` : CausalGraph
- re (standard library)

No other CausalLens components are imported.
"""

from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from causallens.types import CausalGraph


class QueryType(Enum):
    """Type of causal query."""
    COUNTERFACTUAL = "counterfactual"   # "What would Y be if X was 5?"
    ATE = "ate"                         # "What is the effect of X on Y?"
    CAUSAL_CHECK = "causal_check"       # "Does X cause Y?"


class InterventionType(Enum):
    """How the intervention value should be interpreted."""
    SET = "set"                 # do(X = value)
    INCREASE_BY = "increase"    # do(X = X + value)
    DECREASE_BY = "decrease"    # do(X = X - value)
    MULTIPLY_BY = "multiply"    # do(X = X * value)
    INCREASE_PCT = "increase_pct"   # do(X = X * (1 + pct/100))
    DECREASE_PCT = "decrease_pct"   # do(X = X * (1 - pct/100))


@dataclass
class ParsedQuery:
    """Structured result of parsing a natural language query."""
    query_type: QueryType
    target: Optional[str] = None
    intervention_var: Optional[str] = None
    intervention_value: Optional[float] = None
    intervention_type: InterventionType = InterventionType.SET
    original_query: str = ""
    confidence: float = 1.0     # 0-1, how confident the parse is
    raw_target: str = ""        # what was matched before variable resolution
    raw_intervention_var: str = ""

    def __repr__(self) -> str:
        if self.query_type == QueryType.COUNTERFACTUAL:
            return (f"ParsedQuery(target='{self.target}', "
                    f"do({self.intervention_var} "
                    f"{self.intervention_type.value} "
                    f"{self.intervention_value}))")
        elif self.query_type == QueryType.ATE:
            return (f"ParsedQuery(ATE of '{self.intervention_var}' "
                    f"on '{self.target}')")
        else:
            return (f"ParsedQuery(does '{self.intervention_var}' "
                    f"cause '{self.target}'?)")


class QueryParser:
    """
    Parse natural language causal queries into structured interventions.

    Parameters
    ----------
    variables : list[str]
        Known variable names (column names from the dataset).
        Used for fuzzy matching against the query.

    Examples
    --------
    >>> parser = QueryParser(["ad_spend", "revenue", "website_traffic"])
    >>> q = parser.parse("What would revenue be if ad spend was 10000?")
    >>> q.target
    'revenue'
    >>> q.intervention_var
    'ad_spend'
    >>> q.intervention_value
    10000.0
    """

    def __init__(self, variables: List[str]) -> None:
        if not variables:
            raise ValueError("variables list cannot be empty")
        self.variables = list(variables)
        # Pre-compute normalized forms for matching
        self._var_normalized = {
            v: v.lower().replace("_", " ").replace("-", " ")
            for v in variables
        }

    @classmethod
    def from_graph(cls, graph: CausalGraph) -> "QueryParser":
        """Create a parser from a CausalGraph's variable names."""
        return cls(graph.variables)

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into a structured ParsedQuery.

        Parameters
        ----------
        query : str
            Natural language causal question.

        Returns
        -------
        ParsedQuery
            Structured representation of the query.

        Raises
        ------
        ValueError
            If the query cannot be parsed.
        """
        query_clean = query.strip()
        if not query_clean:
            raise ValueError("Query is empty")

        # Try each pattern in order of specificity
        result = (
            self._try_causal_check(query_clean)
            or self._try_ate_query(query_clean)
            or self._try_counterfactual_set(query_clean)
            or self._try_counterfactual_change(query_clean)
            or self._try_counterfactual_multiplier(query_clean)
            or self._try_counterfactual_percent(query_clean)
            or self._try_counterfactual_inverted(query_clean)
            or self._try_generic_what_if(query_clean)
        )

        if result is None:
            raise ValueError(
                f"Could not parse query: '{query_clean}'. "
                f"Try patterns like: 'What would Y be if X was 5?' or "
                f"'What happens to Y if we increase X by 10?'"
            )

        result.original_query = query_clean
        return result

    # ── Pattern matchers ─────────────────────────────────────────────────

    def _try_counterfactual_set(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "What would {target} be if {var} was/were (set to) {value}?"
        """
        patterns = [
            # "What would revenue be if ad_spend was 10000"
            r"what\s+would\s+(.+?)\s+be\s+if\s+(.+?)\s+(?:was|were|is|=)\s*(?:set\s+to\s+)?(-?[\d,.]+)",
            # "What is revenue if ad_spend is 10000"
            r"what\s+(?:is|would\s+be)\s+(.+?)\s+(?:if|when)\s+(.+?)\s+(?:is|was|were|=)\s*(?:set\s+to\s+)?(-?[\d,.]+)",
            # "What happens to revenue if ad_spend was 10000"
            r"what\s+(?:happens?|would\s+happen)\s+to\s+(.+?)\s+if\s+(.+?)\s+(?:was|were|is|=)\s*(?:set\s+to\s+)?(-?[\d,.]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, q.lower())
            if m:
                target = self._resolve_variable(m.group(1).strip())
                var = self._resolve_variable(m.group(2).strip())
                value = self._parse_number(m.group(3))
                if target and var and value is not None:
                    return ParsedQuery(
                        query_type=QueryType.COUNTERFACTUAL,
                        target=target,
                        intervention_var=var,
                        intervention_value=value,
                        intervention_type=InterventionType.SET,
                        raw_target=m.group(1).strip(),
                        raw_intervention_var=m.group(2).strip(),
                    )
        return None

    def _try_counterfactual_change(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "What happens to {target} if we increase/decrease {var} by {value}?"
        """
        patterns = [
            r"what\s+(?:happens?|would\s+happen)\s+to\s+(.+?)\s+if\s+(?:we\s+)?(increase|decrease|raise|lower|reduce|boost|cut)\s+(.+?)\s+by\s+(-?[\d,.]+)",
            r"(?:if\s+(?:we\s+)?)(increase|decrease|raise|lower|reduce|boost|cut)\s+(.+?)\s+by\s+(-?[\d,.]+)\s*,?\s*what\s+(?:happens?|would\s+happen)\s+to\s+(.+)",
        ]
        # Pattern 1: target first
        m = re.search(patterns[0], q.lower())
        if m:
            target = self._resolve_variable(m.group(1).strip())
            direction = m.group(2).strip()
            var = self._resolve_variable(m.group(3).strip())
            value = self._parse_number(m.group(4))

            if target and var and value is not None:
                int_type = (InterventionType.INCREASE_BY
                            if direction in ("increase", "raise", "boost")
                            else InterventionType.DECREASE_BY)
                return ParsedQuery(
                    query_type=QueryType.COUNTERFACTUAL,
                    target=target,
                    intervention_var=var,
                    intervention_value=value,
                    intervention_type=int_type,
                    raw_target=m.group(1).strip(),
                    raw_intervention_var=m.group(3).strip(),
                )

        # Pattern 2: direction first
        m = re.search(patterns[1], q.lower())
        if m:
            direction = m.group(1).strip()
            var = self._resolve_variable(m.group(2).strip())
            value = self._parse_number(m.group(3))
            target = self._resolve_variable(m.group(4).strip().rstrip("?"))

            if target and var and value is not None:
                int_type = (InterventionType.INCREASE_BY
                            if direction in ("increase", "raise", "boost")
                            else InterventionType.DECREASE_BY)
                return ParsedQuery(
                    query_type=QueryType.COUNTERFACTUAL,
                    target=target,
                    intervention_var=var,
                    intervention_value=value,
                    intervention_type=int_type,
                    raw_target=m.group(4).strip(),
                    raw_intervention_var=m.group(2).strip(),
                )
        return None

    def _try_counterfactual_multiplier(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "What happens to {target} if we double/triple/halve {var}?"
        """
        multipliers = {
            "double": 2.0, "doubled": 2.0,
            "triple": 3.0, "tripled": 3.0,
            "quadruple": 4.0, "quadrupled": 4.0,
            "halve": 0.5, "halved": 0.5,
        }
        pattern = (
            r"what\s+(?:would|happens?\s+to)\s+(.+?)\s+"
            r"(?:be\s+)?if\s+(?:we\s+)?"
            r"(double[d]?|triple[d]?|quadruple[d]?|halve[d]?)\s+"
            r"(?:the\s+)?(.+?)[\s?]*$"
        )
        m = re.search(pattern, q.lower())
        if m:
            target = self._resolve_variable(m.group(1).strip())
            mult_word = m.group(2).strip()
            var = self._resolve_variable(m.group(3).strip().rstrip("?"))
            mult_value = multipliers.get(mult_word)

            if target and var and mult_value is not None:
                return ParsedQuery(
                    query_type=QueryType.COUNTERFACTUAL,
                    target=target,
                    intervention_var=var,
                    intervention_value=mult_value,
                    intervention_type=InterventionType.MULTIPLY_BY,
                    raw_target=m.group(1).strip(),
                    raw_intervention_var=m.group(3).strip(),
                )
        return None

    def _try_counterfactual_percent(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "How would {target} change if {var} increased/decreased by {pct}%?"
        """
        pattern = (
            r"(?:how\s+would|what\s+happens?\s+to)\s+(.+?)\s+"
            r"(?:change\s+)?if\s+(?:we\s+)?"
            r"(?:(?:increase|decrease|raise|lower|reduce|boost|cut)\s+)?(.+?)\s+"
            r"(?:increase[ds]?|decrease[ds]?|raise[ds]?|lower[ds]?|reduce[ds]?|go(?:es)?\s+(?:up|down))\s+"
            r"(?:by\s+)?(-?[\d,.]+)\s*%"
        )
        m = re.search(pattern, q.lower())
        if m:
            target = self._resolve_variable(m.group(1).strip())
            var = self._resolve_variable(m.group(2).strip())
            pct = self._parse_number(m.group(3))

            if target and var and pct is not None:
                # Determine direction from the verb
                direction_match = re.search(
                    r"(decrease|lower|reduce|go(?:es)?\s+down)", q.lower()
                )
                int_type = (InterventionType.DECREASE_PCT
                            if direction_match
                            else InterventionType.INCREASE_PCT)
                return ParsedQuery(
                    query_type=QueryType.COUNTERFACTUAL,
                    target=target,
                    intervention_var=var,
                    intervention_value=pct,
                    intervention_type=int_type,
                    raw_target=m.group(1).strip(),
                    raw_intervention_var=m.group(2).strip(),
                )
        return None

    def _try_counterfactual_inverted(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "If {var} had been {value}, what would {target} have been?"
        """
        pattern = (
            r"if\s+(.+?)\s+(?:had\s+been|was|were)\s+(-?[\d,.]+)\s*,?\s*"
            r"what\s+would\s+(.+?)\s+(?:have\s+been|be)"
        )
        m = re.search(pattern, q.lower())
        if m:
            var = self._resolve_variable(m.group(1).strip())
            value = self._parse_number(m.group(2))
            target = self._resolve_variable(m.group(3).strip())

            if target and var and value is not None:
                return ParsedQuery(
                    query_type=QueryType.COUNTERFACTUAL,
                    target=target,
                    intervention_var=var,
                    intervention_value=value,
                    intervention_type=InterventionType.SET,
                    raw_target=m.group(3).strip(),
                    raw_intervention_var=m.group(1).strip(),
                )
        return None

    def _try_generic_what_if(self, q: str) -> Optional[ParsedQuery]:
        """
        Fallback: "What if {var} was {value}? → {target}" or simple
        "{var} = {value}, what about {target}?"
        """
        pattern = r"what\s+if\s+(.+?)\s+(?:was|were|is|=)\s*(-?[\d,.]+)\s*\??\s*(?:→|->|,)?\s*(.+?)[\s?]*$"
        m = re.search(pattern, q.lower())
        if m:
            var = self._resolve_variable(m.group(1).strip())
            value = self._parse_number(m.group(2))
            target = self._resolve_variable(m.group(3).strip().rstrip("?"))

            if target and var and value is not None:
                return ParsedQuery(
                    query_type=QueryType.COUNTERFACTUAL,
                    target=target,
                    intervention_var=var,
                    intervention_value=value,
                    intervention_type=InterventionType.SET,
                    raw_target=m.group(3).strip(),
                    raw_intervention_var=m.group(1).strip(),
                )
        return None

    def _try_ate_query(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "What is the effect of {var} on {target}?"
        """
        patterns = [
            r"what\s+is\s+the\s+(?:causal\s+)?effect\s+of\s+(.+?)\s+on\s+(.+?)[\s?]*$",
            r"how\s+(?:much\s+)?does\s+(.+?)\s+(?:affect|influence|impact)\s+(.+?)[\s?]*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, q.lower())
            if m:
                var = self._resolve_variable(m.group(1).strip())
                target = self._resolve_variable(m.group(2).strip().rstrip("?"))
                if var and target:
                    return ParsedQuery(
                        query_type=QueryType.ATE,
                        target=target,
                        intervention_var=var,
                        raw_target=m.group(2).strip(),
                        raw_intervention_var=m.group(1).strip(),
                    )
        return None

    def _try_causal_check(self, q: str) -> Optional[ParsedQuery]:
        """
        Match: "Does {var} cause {target}?"
        """
        patterns = [
            r"does\s+(.+?)\s+cause\s+(.+?)[\s?]*$",
            r"is\s+(.+?)\s+a\s+cause\s+of\s+(.+?)[\s?]*$",
            r"(?:is\s+there\s+)?(?:a\s+)?causal\s+(?:relationship|link|connection)\s+(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)[\s?]*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, q.lower())
            if m:
                var = self._resolve_variable(m.group(1).strip())
                target = self._resolve_variable(m.group(2).strip().rstrip("?"))
                if var and target:
                    return ParsedQuery(
                        query_type=QueryType.CAUSAL_CHECK,
                        target=target,
                        intervention_var=var,
                        raw_target=m.group(2).strip(),
                        raw_intervention_var=m.group(1).strip(),
                    )
        return None

    # ── Variable resolution ──────────────────────────────────────────────

    def _resolve_variable(self, text: str) -> Optional[str]:
        """
        Fuzzy-match a text fragment against known variable names.

        Priority: exact match > normalized match > partial match.
        Returns None if no match found.
        """
        text_clean = text.lower().strip().rstrip("?.,!")

        # 1. Exact match
        for v in self.variables:
            if v.lower() == text_clean:
                return v

        # 2. Normalized match (spaces ↔ underscores)
        text_norm = text_clean.replace("_", " ").replace("-", " ")
        for v, v_norm in self._var_normalized.items():
            if v_norm == text_norm:
                return v

        # 3. Partial match — text contains variable name or vice versa
        # Prefer longer matches (more specific)
        candidates = []
        for v, v_norm in self._var_normalized.items():
            if v_norm in text_norm or text_norm in v_norm:
                candidates.append((v, len(v_norm)))

        if candidates:
            # Return the longest match (most specific)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # 4. Word overlap — any word in text matches any word in variable
        text_words = set(text_norm.split())
        best_match = None
        best_overlap = 0
        for v, v_norm in self._var_normalized.items():
            v_words = set(v_norm.split())
            overlap = len(text_words & v_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = v

        if best_overlap > 0:
            return best_match

        return None

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """Parse a number from text, handling commas and decimals."""
        try:
            cleaned = text.replace(",", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return None
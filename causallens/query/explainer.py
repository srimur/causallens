"""
causallens.query.explainer
============================
Explainability layer for causal queries.

Two responsibilities:
1. **Parse explanation** — show the user how their natural language
   question was decomposed into a structured intervention.
2. **Reasoning explanation** — given a causal result, explain the
   chain of causal reasoning that produced the answer.

This is useful for:
- Debugging unexpected query results
- Building trust in the causal model's output
- Understanding which variables mediated the effect
- Identifying which assumptions the answer depends on

Dependencies
------------
- ``causallens.types`` : CausalGraph, EdgeType
- ``causallens.query.parser`` : ParsedQuery, QueryType, InterventionType
- ``causallens.graph.visualize`` : find_causal_path

No other CausalLens components are imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from causallens.types import CausalGraph, EdgeType
from causallens.query.parser import ParsedQuery, QueryType, InterventionType
from causallens.graph.visualize import find_causal_path


@dataclass
class ParseExplanation:
    """Structured explanation of how a query was parsed."""
    original_query: str
    query_type: str
    query_type_description: str
    intervention_var: Optional[str]
    intervention_var_matched_from: Optional[str]
    intervention_type: Optional[str]
    intervention_type_description: Optional[str]
    intervention_value: Optional[float]
    target_var: Optional[str]
    target_var_matched_from: Optional[str]
    do_notation: Optional[str]
    confidence: float
    steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        lines = ["Parse Explanation", "=" * 40]
        lines.append(f'  Input:    "{self.original_query}"')
        lines.append(f"  Type:     {self.query_type} — {self.query_type_description}")
        if self.intervention_var:
            lines.append(f"  Cause:    {self.intervention_var} (matched from \"{self.intervention_var_matched_from}\")")
        if self.target_var:
            lines.append(f"  Effect:   {self.target_var} (matched from \"{self.target_var_matched_from}\")")
        if self.do_notation:
            lines.append(f"  Formal:   {self.do_notation}")
        if self.intervention_type_description:
            lines.append(f"  Action:   {self.intervention_type_description}")
        lines.append(f"  Confidence: {self.confidence:.0%}")
        lines.append("")
        lines.append("  Reasoning steps:")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"    {i}. {step}")
        return "\n".join(lines)


@dataclass
class CausalExplanation:
    """Structured explanation of the causal reasoning behind a result."""
    query_type: str
    summary: str
    causal_path: Optional[List[str]]
    path_description: Optional[str]
    mediators: List[str]
    mechanism_steps: List[str]
    assumptions: List[str]
    edge_contributions: List[Dict[str, Any]]
    total_effect: Optional[float]
    confidence_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        lines = ["Causal Explanation", "=" * 40]
        lines.append(f"  {self.summary}")
        lines.append("")
        if self.causal_path:
            lines.append(f"  Path: {' → '.join(self.causal_path)}")
        if self.mediators:
            lines.append(f"  Mediators: {', '.join(self.mediators)}")
        lines.append("")
        lines.append("  Mechanism:")
        for i, step in enumerate(self.mechanism_steps, 1):
            lines.append(f"    {i}. {step}")
        if self.edge_contributions:
            lines.append("")
            lines.append("  Edge contributions:")
            for ec in self.edge_contributions:
                lines.append(f"    {ec['from']} → {ec['to']}: coefficient = {ec['coefficient']}")
        if self.total_effect is not None:
            lines.append(f"  Total path effect: {self.total_effect:.6f}")
        lines.append("")
        lines.append("  Assumptions:")
        for a in self.assumptions:
            lines.append(f"    • {a}")
        return "\n".join(lines)


class QueryExplainer:
    """
    Generate explanations for how queries are parsed and answered.

    Parameters
    ----------
    graph : CausalGraph
        The discovered causal graph.
    coefficients : dict or None
        Variable → {parent: coefficient} mapping from the fitted model.
        If provided, enables quantitative path explanations.

    Examples
    --------
    >>> explainer = QueryExplainer(graph, engine.get_all_coefficients())
    >>> parse_exp = explainer.explain_parse(parsed_query)
    >>> print(parse_exp)
    >>> causal_exp = explainer.explain_result(parsed_query, result)
    >>> print(causal_exp)
    """

    QUERY_TYPE_DESCRIPTIONS = {
        QueryType.COUNTERFACTUAL: "Counterfactual — asks what would happen under a hypothetical intervention",
        QueryType.ATE: "Average Treatment Effect — asks for the population-level causal effect",
        QueryType.CAUSAL_CHECK: "Causal Check — asks whether a causal path exists between two variables",
    }

    INTERVENTION_TYPE_DESCRIPTIONS = {
        InterventionType.SET: "Set variable to a specific value (do-operator)",
        InterventionType.INCREASE_BY: "Increase variable by an absolute amount",
        InterventionType.DECREASE_BY: "Decrease variable by an absolute amount",
        InterventionType.MULTIPLY_BY: "Multiply variable by a factor",
        InterventionType.INCREASE_PCT: "Increase variable by a percentage",
        InterventionType.DECREASE_PCT: "Decrease variable by a percentage",
    }

    def __init__(
        self,
        graph: CausalGraph,
        coefficients: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.graph = graph
        self.coefficients = coefficients or {}

    def explain_parse(self, parsed: ParsedQuery) -> ParseExplanation:
        """
        Explain how a natural language query was parsed.

        Shows the matched variables, query type classification,
        intervention decomposition, and formal do-notation.
        """
        steps = []

        # Step 1: Classification
        steps.append(
            f'Classified as {parsed.query_type.value} query based on keyword patterns'
        )

        # Step 2: Variable matching
        if parsed.raw_intervention_var and parsed.intervention_var:
            if parsed.raw_intervention_var == parsed.intervention_var:
                steps.append(f'Matched cause variable: "{parsed.raw_intervention_var}" → exact match')
            else:
                steps.append(
                    f'Matched cause variable: "{parsed.raw_intervention_var}" '
                    f'→ fuzzy-matched to "{parsed.intervention_var}" '
                    f'(underscore/case normalization or substring match)'
                )

        if parsed.raw_target and parsed.target:
            if parsed.raw_target == parsed.target:
                steps.append(f'Matched target variable: "{parsed.raw_target}" → exact match')
            else:
                steps.append(
                    f'Matched target variable: "{parsed.raw_target}" '
                    f'→ fuzzy-matched to "{parsed.target}"'
                )

        # Step 3: Intervention parsing
        if parsed.query_type == QueryType.COUNTERFACTUAL:
            steps.append(
                f'Parsed intervention: {parsed.intervention_type.value}'
                f'({parsed.intervention_var} = {parsed.intervention_value})'
            )

        # Step 4: Do-notation
        do_notation = self._to_do_notation(parsed)
        if do_notation:
            steps.append(f'Formal representation: {do_notation}')

        return ParseExplanation(
            original_query=parsed.original_query,
            query_type=parsed.query_type.value,
            query_type_description=self.QUERY_TYPE_DESCRIPTIONS.get(
                parsed.query_type, "Unknown"
            ),
            intervention_var=parsed.intervention_var,
            intervention_var_matched_from=parsed.raw_intervention_var or parsed.intervention_var,
            intervention_type=parsed.intervention_type.value if parsed.intervention_type else None,
            intervention_type_description=self.INTERVENTION_TYPE_DESCRIPTIONS.get(
                parsed.intervention_type
            ),
            intervention_value=parsed.intervention_value,
            target_var=parsed.target,
            target_var_matched_from=parsed.raw_target or parsed.target,
            do_notation=do_notation,
            confidence=parsed.confidence,
            steps=steps,
        )

    def explain_result(
        self,
        parsed: ParsedQuery,
        result: Dict[str, Any],
    ) -> CausalExplanation:
        """
        Explain the causal reasoning behind a query result.

        Shows the causal path, mediating variables, per-edge contributions,
        mechanism description, and the assumptions the answer depends on.
        """
        if parsed.query_type == QueryType.COUNTERFACTUAL:
            return self._explain_counterfactual(parsed, result)
        elif parsed.query_type == QueryType.ATE:
            return self._explain_ate(parsed, result)
        elif parsed.query_type == QueryType.CAUSAL_CHECK:
            return self._explain_causal_check(parsed, result)
        else:
            return CausalExplanation(
                query_type=parsed.query_type.value,
                summary="Unknown query type",
                causal_path=None, path_description=None,
                mediators=[], mechanism_steps=[], assumptions=[],
                edge_contributions=[], total_effect=None,
                confidence_factors=[],
            )

    def _explain_counterfactual(
        self, parsed: ParsedQuery, result: Dict[str, Any]
    ) -> CausalExplanation:
        iv = parsed.intervention_var
        target = parsed.target
        path = find_causal_path(self.graph, iv, target) if iv and target else None

        mediators = path[1:-1] if path and len(path) > 2 else []
        mechanism_steps = []
        edge_contribs = []
        total_effect = None

        if path:
            # Step 1: Intervention
            mechanism_steps.append(
                f'Apply do({iv} = {parsed.intervention_value}) — '
                f'this severs all incoming edges to {iv} and forces its value'
            )

            # Steps for each edge in the path
            for i in range(len(path) - 1):
                src, tgt = path[i], path[i + 1]
                coef = self._get_coefficient(src, tgt)
                if coef is not None:
                    mechanism_steps.append(
                        f'{src} → {tgt}: structural equation coefficient = {coef:+.4f}. '
                        f'A unit change in {src} causes a {abs(coef):.4f} unit '
                        f'{"increase" if coef > 0 else "decrease"} in {tgt}.'
                    )
                    edge_contribs.append({
                        "from": src, "to": tgt, "coefficient": coef
                    })
                else:
                    mechanism_steps.append(
                        f'{src} → {tgt}: directed edge exists (coefficient not available)'
                    )

            # Total effect
            if edge_contribs:
                total_effect = 1.0
                for ec in edge_contribs:
                    total_effect *= ec["coefficient"]
                mechanism_steps.append(
                    f'Total path effect (product of coefficients): {total_effect:.6f}'
                )

            # Abduction step
            mechanism_steps.append(
                f'Noise terms (ε) for non-intervened variables are preserved from '
                f'the observed data (abduction step) to maintain individual-level consistency'
            )

        effect = result.get("effect", 0)
        factual = result.get("factual_value", 0)
        cf = result.get("counterfactual_value", 0)

        summary = (
            f'Under do({iv} = {parsed.intervention_value}), {target} changes '
            f'from {factual:.2f} to {cf:.2f} (effect: {effect:+.2f}). '
        )
        if mediators:
            summary += f'The effect propagates through: {" → ".join(mediators)}.'

        path_desc = None
        if path:
            path_desc = " → ".join(path)

        return CausalExplanation(
            query_type="counterfactual",
            summary=summary,
            causal_path=path,
            path_description=path_desc,
            mediators=mediators,
            mechanism_steps=mechanism_steps,
            assumptions=self._standard_assumptions(iv, target, path),
            edge_contributions=edge_contribs,
            total_effect=total_effect,
            confidence_factors=self._confidence_factors(path),
        )

    def _explain_ate(
        self, parsed: ParsedQuery, result: Dict[str, Any]
    ) -> CausalExplanation:
        iv = parsed.intervention_var
        target = parsed.target
        path = find_causal_path(self.graph, iv, target) if iv and target else None
        mediators = path[1:-1] if path and len(path) > 2 else []

        ate = result.get("ate", 0)
        n = result.get("n_samples", 0)

        mechanism_steps = [
            f'For each of the {n} observations, compute the counterfactual '
            f'outcome under treatment vs control',
            f'ATE = mean(Y_treated − Y_control) across all {n} samples',
            f'Result: ATE = {ate:.4f}',
        ]

        return CausalExplanation(
            query_type="ate",
            summary=f'Average Treatment Effect of {iv} on {target}: {ate:+.4f} (n={n})',
            causal_path=path,
            path_description=" → ".join(path) if path else None,
            mediators=mediators,
            mechanism_steps=mechanism_steps,
            assumptions=self._standard_assumptions(iv, target, path),
            edge_contributions=[],
            total_effect=ate,
            confidence_factors=self._confidence_factors(path),
        )

    def _explain_causal_check(
        self, parsed: ParsedQuery, result: Dict[str, Any]
    ) -> CausalExplanation:
        iv = parsed.intervention_var
        target = parsed.target
        has_path = result.get("has_causal_path", False)
        is_direct = result.get("is_direct_cause", False)
        path = result.get("causal_path", None)

        if is_direct:
            summary = f'{iv} is a direct cause of {target} (directed edge in the graph).'
            mechanism_steps = [
                f'Found directed edge: {iv} → {target}',
                f'This means {iv} appears as a parent in the structural equation for {target}',
            ]
        elif has_path:
            summary = f'{iv} causes {target} indirectly through a directed path.'
            mechanism_steps = [
                f'No direct edge {iv} → {target}',
                f'Found directed path: {" → ".join(path)}',
                f'The effect propagates through mediating variables',
            ]
        else:
            summary = f'No causal path from {iv} to {target} in the discovered graph.'
            mechanism_steps = [
                f'No directed path found from {iv} to {target}',
                f'This means the PC algorithm found no chain of conditional '
                f'dependencies connecting these variables',
            ]

        return CausalExplanation(
            query_type="causal_check",
            summary=summary,
            causal_path=path,
            path_description=" → ".join(path) if path else None,
            mediators=path[1:-1] if path and len(path) > 2 else [],
            mechanism_steps=mechanism_steps,
            assumptions=[
                "Causal Markov condition holds",
                "Faithfulness: all conditional independencies reflect the true graph",
                "No latent confounders (causal sufficiency)",
            ],
            edge_contributions=[],
            total_effect=None,
            confidence_factors=[],
        )

    def _get_coefficient(self, source: str, target: str) -> Optional[float]:
        """Look up the structural equation coefficient for source → target."""
        if target in self.coefficients:
            return self.coefficients[target].get(source)
        return None

    def _to_do_notation(self, parsed: ParsedQuery) -> Optional[str]:
        """Convert a parsed query to formal do-calculus notation."""
        if parsed.query_type == QueryType.COUNTERFACTUAL:
            iv = parsed.intervention_var
            val = parsed.intervention_value
            target = parsed.target
            return f'P({target} | do({iv} = {val}))'
        elif parsed.query_type == QueryType.ATE:
            iv = parsed.intervention_var
            target = parsed.target
            return f'E[{target} | do({iv} = 1)] − E[{target} | do({iv} = 0)]'
        elif parsed.query_type == QueryType.CAUSAL_CHECK:
            return f'∃ directed path {parsed.intervention_var} ⇝ {parsed.target}?'
        return None

    def _standard_assumptions(
        self, source: Optional[str], target: Optional[str],
        path: Optional[List[str]],
    ) -> List[str]:
        """Return the assumptions that the causal answer depends on."""
        assumptions = [
            "Causal Markov condition: each variable is independent of its "
            "non-descendants given its parents",
            "Faithfulness: all conditional independencies in the data reflect "
            "the true causal structure",
            "Causal sufficiency: no unmeasured common causes (latent confounders)",
            "Acyclicity: the true causal graph is a DAG",
        ]
        if path and len(path) > 2:
            mediators = path[1:-1]
            assumptions.append(
                f'No unmodeled interaction effects between mediators '
                f'({", ".join(mediators)})'
            )
        return assumptions

    def _confidence_factors(self, path: Optional[List[str]]) -> List[str]:
        """List factors that affect confidence in the result."""
        factors = []
        if path:
            if len(path) > 3:
                factors.append(
                    f'Long causal path ({len(path)} nodes) — effect estimates '
                    f'compound errors at each step'
                )
            if len(path) == 2:
                factors.append('Direct causal link — fewer sources of error')
        return factors

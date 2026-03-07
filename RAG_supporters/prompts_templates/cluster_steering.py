"""Prompt templates for cluster steering and question rephrasing."""

CLUSTER_ACTIVATION_PROMPT = """Given a question and a target cluster/topic with its descriptors, generate a steering text that would help activate this specific subspace in a RAG system.

QUESTION:
{question}

TARGET CLUSTER ID: {cluster_id}

CLUSTER DESCRIPTORS:
{cluster_descriptors}

Requirements:
- Generate a concise steering text (1-3 sentences) that emphasizes the target cluster's themes
- The steering text should guide the retrieval toward this specific knowledge domain
- Incorporate relevant descriptor keywords naturally
- Maintain semantic coherence with the original question
- The text should be suitable for embedding and similarity search

Output the result in the following JSON format:
{{
  "cluster_id": {cluster_id},
  "steering_text": "Generated steering text that activates the target cluster",
  "incorporated_descriptors": ["descriptor1", "descriptor2"],
  "confidence": 0.85
}}

Provide only the JSON output, no additional text."""


QUESTION_REPHRASE_PROMPT = """Given a question and target cluster information, rephrase the question to steer it toward a specific genre/topic while preserving the core intent.

ORIGINAL QUESTION:
{question}

TARGET CLUSTER ID: {cluster_id}

CLUSTER DESCRIPTORS:
{cluster_descriptors}

ALTERNATE CLUSTER IDs: {alternate_clusters}

Requirements:
- Rephrase the question to emphasize the target cluster's domain
- Preserve the core information need of the original question
- Use vocabulary and phrasing typical of the target domain
- Generate variations that could activate different subspaces for multi-membership cases
- Ensure the rephrased question is natural and coherent

Output the result in the following JSON format:
{{
  "original_question": "Original question text",
  "target_cluster_id": {cluster_id},
  "rephrased_question": "Rephrased version emphasizing target cluster",
  "genre_shift": "Description of how genre/topic shifted",
  "preserved_intent": "Core information need that was preserved",
  "confidence": 0.9
}}

Provide only the JSON output, no additional text."""


MULTI_CLUSTER_REPHRASE_PROMPT = """Given a question that spans multiple topics/clusters, generate variations that emphasize different aspects to activate specific subspaces.

ORIGINAL QUESTION:
{question}

CANDIDATE CLUSTERS:
{cluster_info}

Requirements:
- Generate {num_variations} question variations, each emphasizing a different cluster
- Each variation should preserve the original intent but shift focus to a specific domain
- Variations should be natural and coherent
- Use cluster descriptors to guide the rephrasing

Output the result in the following JSON format:
{{
  "original_question": "Original question text",
  "variations": [
    {{
      "target_cluster_id": 0,
      "rephrased_question": "Variation emphasizing cluster 0",
      "emphasized_aspect": "What aspect is emphasized",
      "incorporated_descriptors": ["descriptor1", "descriptor2"]
    }}
  ],
  "total_variations": {num_variations}
}}

Provide only the JSON output, no additional text."""


AMBIGUITY_RESOLUTION_PROMPT = """Given a question and cluster assignments with probabilities, analyze the ambiguity and suggest the most relevant clusters.

QUESTION:
{question}

CLUSTER ASSIGNMENTS:
{cluster_assignments}

Requirements:
- Analyze why the question might be relevant to multiple clusters
- Rank clusters by relevance to the question's core intent
- Identify the primary topic and secondary topics
- Suggest whether this is a multi-domain question or has one clear primary domain
- Provide confidence scores for cluster relevance

Output the result in the following JSON format:
{{
  "question": "The analyzed question",
  "is_ambiguous": true,
  "primary_cluster": {{
    "cluster_id": 0,
    "reason": "Why this is the primary cluster",
    "confidence": 0.85
  }},
  "secondary_clusters": [
    {{
      "cluster_id": 1,
      "reason": "Why this cluster is also relevant",
      "confidence": 0.45
    }}
  ],
  "recommendation": "single-domain|multi-domain",
  "explanation": "Overall analysis of the question's domain coverage"
}}

Provide only the JSON output, no additional text."""

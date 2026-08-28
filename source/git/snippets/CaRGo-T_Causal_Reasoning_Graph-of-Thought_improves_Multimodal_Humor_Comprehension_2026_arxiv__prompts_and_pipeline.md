# CaRGo-T: Prompt and Pipeline Implementation Snippet

> 원본 코드: [`source/git/CaRGo-T_abhi1nandy2/src/yesbut_generation.py`](../source/git/CaRGo-T_abhi1nandy2/src/yesbut_generation.py), [`source/git/CaRGo-T_abhi1nandy2/src/yesbut_detection.py`](../source/git/CaRGo-T_abhi1nandy2/src/yesbut_detection.py), [`source/git/CaRGo-T_abhi1nandy2/src/utils.py`](../source/git/CaRGo-T_abhi1nandy2/src/utils.py)

---

## 1. 프롬프트 템플릿 및 변형 정의 (`src/yesbut_generation.py`)

CaRGo-T는 원본 과업 질문(Task-Specific Query)에 인과 추론 그래프(Causal Reasoning Graph, CRG) 생성을 강제하는 지시문을 결합하여 모델이 최종 응답 전에 명시적 인과 구조를 코드(JSON) 형태로 출력하도록 유도한다.

```python
causal_graph_description = """
1. Entities: There is a set of entities (listed in "entities") — each entity can have properties that are either descriptions of/adjectives/adverbs qualifying that entity or (non-causal) relations with other entities. These are listed under “properties” attribute of each entity.
e.g entity: ANIMAL, "properties": ["Cats and Dogs", “pet of HUMAN”, “hides under FURNITURE”], note that HUMAN AND FURNITURE are other entities
entity: FIREWORKS, "properties": [ "Bright and colorful explosions in sky", "burnt by HUMAN”] 
note that the (non—causal) relationships are bidirectional, e.g. FIRECRACKERS (burnt by) HUMAN, HUMAN (burns) FIRECRACKER are same relationships. This relationship is present in "properties" list of any ONE of these entities e.g ("burns FIRECRACKERS" belongs to HUMAN[“properties"]) OR ("burnt by HUMAN" belongs to FIRECRACKER[“properties"]), BUT NOT BOTH 

2. CAUSAL relationships: listed under "causal_relationships". First, we define an EVENT. A collection of entities (along with their (non—causal) relationships) describes an EVENT which is typically of the form "X (optionally) does Y (optionally) with/for/to Z", (a single entity can also be an EVENT) — an EVENT is basically a macro node and a causal relation is defined between events.
A causal relation is listed under "causal_relationships" as a dictionary {"cause": EVENT_1, "effect': EVENT_2}. Each event is expressed in natural language which tells what the collection of entities means, for instance, “X (optionally) does Y (optionally) with/for/to Z”. For example, {"cause": “HUMAN burns FIRECRACKER S”, "effect': “ANIMALS” are frightened}
"""

prompt_vanilla = "Why is this image funny/satirical?"

prompt_vanilla_cot = (
    "Why is this image funny/satirical? Analyze the images, their entities and interactions very carefully. "
    "Think step by step. Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' "
)

prompt_with_graph = (
    "Why is this image funny/satirical? To answer this, first create a causal reasoning graph linking different "
    "objects, people, and entities present in the image in the form of a piece of code, and then give the final answer. "
    "Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' "
)

prompt_with_kg = (
    "You are a helpful assistant skilled at logical reasoning and detecting elements of humor in images. "
    "Reason carefully about the entities and their interactions in the image and explain why the image is ironic. "
    "To accomplish the task, first extract a list of triplets (which convey an element of humor, similar to that in a knowledge graph) "
    "of the form (<HEAD>, <RELATION>, <TAIL>), where <HEAD> or <TAIL> is an object, person, or entity present in the image, "
    "and <RELATION> shows relation between the <HEAD> and <TAIL>, and then give the final answer. "
    "Start the triples with <<TRIPLES> and end the triples with <</TRIPLES>> . "
    "Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' "
)

prompt_with_graph_and_desc = (
    "Why is this image funny/satirical? To answer this, first create a causal reasoning graph linking different objects, "
    "people, and entities present in the image in the form of a piece of code, and then give the final answer. "
    + "Use the following definition of causal reasoning graph as a guideline: " + causal_graph_description 
    + "Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' "
)

prompt_with_cod = (
    "Why is this image funny/satirical? Analyze the images, their entities and interactions very carefully. "
    "Think step by step, but only keep a minimum draft for each INTERMEDIATE thinking step, with 5 words at most. "
    "Using the thinking steps, generate the final explanation. "
    "Make sure that the final answer/generation is followed after 'FinalAnswerWithoutCode:' "
)

prompt_with_nl_rationale = """Why is this image funny/satirical? Analyze the images, their entities and interactions very carefully. Think step by step and answer the following: 
Subject: short subject/topic in 4 to 5 words
Premise: A single sentence premise about the humor/satire in the image
Punchline: A conclusive sentence about the humor/satire in the image
Irony: The underlying irony/humor/satire described in the image
Satirical Commentary: A commentary in few sentences about the humor/satire in the image.
Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' """
```

---

## 2. In-Context 예시 조립 및 VLM 호출 파이프라인

```python
def getFewShotExamples(nShot, few_shot_examples, prompt_mode):
    assert nShot <= len(few_shot_examples)
    few_shot_prompt = []
    for i in range(nShot):
        image_enc = base64.b64encode(requests.get(few_shot_examples[i][0]).content).decode("utf-8")
        few_shot_prompt.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_map[prompt_mode]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_enc}",
                        "detail": "high",
                    },
                },
            ],
        })
        few_shot_prompt.append({
            "role": "assistant",
            "content": (
                f"{few_shot_examples[i][1]}\nFinal_Answer_{prompt_mode}: {few_shot_examples[i][2]}"
                if prompt_mode != 'vanilla'
                else f"Final_Answer_{prompt_mode}: {few_shot_examples[i][2]}"
            )
        })
    return few_shot_prompt

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def getResponse(client, model, image, nShot, prompt_mode, few_shot_examples):
    nShotInput = getFewShotExamples(nShot, few_shot_examples, prompt_mode)
    base64_image = pil_image_to_base64(image)
    
    model_inp = nShotInput + [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_map[prompt_mode]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=model_inp,
        max_tokens=1000,
    )
    return response.choices[0].message.content
```

---

## 3. 정제된 인과 추론 그래프 JSON 스키마 (`templates/yesbut_causal_graph.jsonl`)

```json
{
  "entities": {
    "high_heels": {
      "properties": ["A fashionable shoe with a tall heel."]
    },
    "feet": {
      "properties": ["wears high heel", "bandages applied at places"]
    },
    "fashion": {
      "properties": ["Cultural or social drive to look stylish or attractive."]
    },
    "discomfort": {
      "properties": ["painful", "negative_physical_effects_on_feet"]
    }
  },
  "causal_relationships": [
    {"cause": "fashion", "effect": "feet wears high heels"},
    {"cause": "feet wears high heels", "effect": "discomfort"}
  ]
}
```

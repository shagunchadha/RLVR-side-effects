"""Off-domain commonsense reasoning probes (10 items, commonsense_qa).

First 10 items of the tau/commonsense_qa validation split, stored statically
so this module has no dependency on the `datasets` library or network access.
Prompt construction is copied verbatim from the notebook's
`define_reasoning_probes` cell. The final answer must appear on its own last
line as "ANSWER: X" (X = choice letter).
"""

COMMONSENSE_QA_ITEMS = [{'idx': 0,
  'question': 'A revolving door is convenient for two direction travel, but it also serves as a '
              'security measure at a what?',
  'choices': {'A': 'bank', 'B': 'library', 'C': 'department store', 'D': 'mall', 'E': 'new york'},
  'answerKey': 'A'},
 {'idx': 1,
  'question': 'What do people aim to do at work?',
  'choices': {'A': 'complete job',
              'B': 'learn from each other',
              'C': 'kill animals',
              'D': 'wear hats',
              'E': 'talk to each other'},
  'answerKey': 'A'},
 {'idx': 2,
  'question': 'Where would you find magazines along side many other printed works?',
  'choices': {'A': 'doctor',
              'B': 'bookstore',
              'C': 'market',
              'D': 'train station',
              'E': 'mortuary'},
  'answerKey': 'B'},
 {'idx': 3,
  'question': 'Where are  you likely to find a hamburger?',
  'choices': {'A': 'fast food restaurant',
              'B': 'pizza',
              'C': 'ground up dead cows',
              'D': 'mouth',
              'E': 'cow carcus'},
  'answerKey': 'A'},
 {'idx': 4,
  'question': 'James was looking for a good place to buy farmland.  Where might he look?',
  'choices': {'A': 'midwest',
              'B': 'countryside',
              'C': 'estate',
              'D': 'farming areas',
              'E': 'illinois'},
  'answerKey': 'A'},
 {'idx': 5,
  'question': 'What island country is ferret popular?',
  'choices': {'A': 'own home',
              'B': 'north carolina',
              'C': 'great britain',
              'D': 'hutch',
              'E': 'outdoors'},
  'answerKey': 'C'},
 {'idx': 6,
  'question': 'In what Spanish speaking North American country can you get a great cup of coffee?',
  'choices': {'A': "mildred's coffee shop",
              'B': 'mexico',
              'C': 'diner',
              'D': 'kitchen',
              'E': 'canteen'},
  'answerKey': 'B'},
 {'idx': 7,
  'question': 'What do animals do when an enemy is approaching?',
  'choices': {'A': 'feel pleasure',
              'B': 'procreate',
              'C': 'pass water',
              'D': 'listen to each other',
              'E': 'sing'},
  'answerKey': 'D'},
 {'idx': 8,
  'question': 'Reading newspaper one of many ways to practice your what?',
  'choices': {'A': 'literacy',
              'B': 'knowing how to read',
              'C': 'money',
              'D': 'buying',
              'E': 'money bank'},
  'answerKey': 'A'},
 {'idx': 9,
  'question': 'What do people typically do while playing guitar?',
  'choices': {'A': 'cry',
              'B': 'hear sounds',
              'C': 'singing',
              'D': 'arthritis',
              'E': 'making music'},
  'answerKey': 'C'}]


def format_commonsense_item(item):
    options = "\n".join(f"{label}. {text}" for label, text in item["choices"].items())
    prompt = (
        f"{item['question']}\n{options}\n"
        "Answer with the letter of the correct choice, then a one-sentence "
        "justification. Finish your reply with the final answer alone on its "
        "own last line, in the exact format: ANSWER: X (where X is the letter "
        "of the correct choice)."
    )
    return {
        "id": f"csqa_{item['idx']}",
        "probe_type": "reasoning",
        "answer_format": "letter",
        "source_idx": item["idx"],
        "question": item["question"],
        "prompt": prompt,
        "ground_truth": item["answerKey"],
    }


COMMONSENSE_QA_PROBES = [format_commonsense_item(item) for item in COMMONSENSE_QA_ITEMS]

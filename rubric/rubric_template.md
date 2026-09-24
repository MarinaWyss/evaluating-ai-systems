# Rubric: [failure mode name]

Version: 0.1
Owner (the person who makes the final call): [name]

## Definition

One sentence, written in terms you can observe in the trace.

## PASS if

- ...

## FAIL if

- ...

## Edge cases

- What if the answer has two problems?
- What if it's partly right?
- Typos, abbreviations, and near-synonyms: ...

## Examples (at least two per label, including borderline ones)

| User query | What the bot did | Label | Why |
|---|---|---|---|
| | | PASS | |
| | | PASS (borderline) | |
| | | FAIL | |
| | | FAIL (borderline) | |

## Ignore

What reviewers should NOT penalize under this rubric (for example tone, length, or formatting).

## Changelog

- 0.1: first draft

---

# Worked example: Assumes the device

Version: 1.0
Owner: [name]

## Definition

The bot gives device-specific advice before the customer has made clear which BeefCake product they mean.

## PASS if

- The customer names the product, even with a typo ("beefcak bel").
- Only one BeefCake product fits what the customer described (only the Row has a screen; only the Bell is charged over USB-C).
- The bot asks which product before giving device-specific steps.

## FAIL if

- The customer's message fits more than one product and the bot picks one without asking.

## Edge cases

- "My heart rate thing keeps dropping": PASS. Team decision: the Pulse is our only heart-rate product, even though the Row's screen can show heart rate.
- General advice that applies to every product (for example "contact BeefCake Support") is not device-specific.

## Examples

| User query | What the bot did | Label | Why |
|---|---|---|---|
| "it won't turn on" | Asked which device | PASS | Asked before advising |
| "my beefcak bel wont click to 12kg" | Answered about the Bell | PASS | Typo, but clearly the Bell |
| "my heart rate thing keeps dropping" | Answered about the Pulse | PASS (borderline) | Team decision, see edge cases |
| "how do I reset it?" | Gave Row factory reset steps | FAIL | "It" could be any device |
| "the battery dies really fast" | Gave Pulse battery steps | FAIL (borderline) | The Bell has a battery too |

## Ignore

Whether the advice itself is correct. That's a different failure mode ("Made-up steps").

## Changelog

- 1.0: added the heart-rate edge case after reviewers disagreed on it
- 0.1: first draft

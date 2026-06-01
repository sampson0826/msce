# Tweet Copy - LeWorldModel Architectural Review

## English (primary)

LeCun says language models aren't the path to AGI. World models are.

He's right about the direction. But look under the hood of his own LeWorldModel:

1. Fixed latent capacity — can't grow to accommodate new patterns
2. Train-once structure — can't prune or restructure through use  
3. No cross-modal phase calibration — can't tell if "seeing a cup lift" and "hearing 'I'll drink'" refer to the same event

LeWorldModel and LLMs are the same architectural genus: fixed capacity, fixed structure, train once. Only difference: text → pixels.

A real world model needs self-restructuring. LeCun's doesn't have it.

---

## Short version (for image alt-text / first tweet)

LeCun: "LLMs aren't the path to AGI. World models are."
Also LeCun: builds a world model with the same architectural genus as LLMs — fixed capacity, fixed structure, train once.

3 structural defects LeWorldModel shares with the systems it critiques.

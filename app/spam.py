from app.model_loader import load_model
def check_spam_rules(text: str) -> str:
  text = text.lower().strip()
  if text == "":
    return "ham", 0
  
  spam_keywords = [
    "free", "win", "winner", "prize", "click",
    "buy now", "urgent", "cash", "money", "offer", "deal",
    "bonus", "limited", "guarantee"
  ]
  
  hit = 0
  for kw in spam_keywords:
    print(kw, text)
    if kw in text:
      hit += 1
  return "spam" if hit >= 2 else "ham", hit

def check_spam_ml(text: str):
  text = text.strip()
  if text == "":
    return "ham", 0.0
  model = load_model()

  pred = model.predict([text])[0]
  proba = model.predict_proba([text])[0]

  classes = list(model.classes_)
  pred_index = classes.index(pred)
  score = float(proba[pred_index])
  
  return pred, score
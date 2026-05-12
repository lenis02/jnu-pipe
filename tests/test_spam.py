from app.spam import check_spam_rules

def test_check_spam_basic():
  assert check_spam_rules("hello")[0] == "ham"
  assert check_spam_rules("free prize click")[0] == "spam"
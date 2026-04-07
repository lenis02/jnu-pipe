from app.spam import check_spam

def test_check_spam_basic():
  assert check_spam("hello")[0] == "ham"
  assert check_spam("free prize click")[0] == "spam"
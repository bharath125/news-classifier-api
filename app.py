from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time

app = Flask(__name__)

model = None
tokenizer = None 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CATEGORIES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
    "comp.windows.x", "misc.forsale", "rec.autos",
    "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
    "sci.crypt", "sci.electronics", "sci.med", "sci.space",
    "soc.religion.christian", "talk.politics.guns",
    "talk.politics.mideast", "talk.politics.misc",
    "talk.religion.misc"
]

def load_model():
  global model, tokenizer
  print("Loading model...")
#   tokenizer = BertTokenizer.from_pretrained('neorange47/news-classifier')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForSequenceClassification.from_pretrained('neorange47/news-classifier')
  model.eval()
  print("Model Loaded")

@app.route('/health',methods=['GET'])
def health():
  return jsonify({
    'status':'ok',
    'model':'news-classifier',
    'timestamp':time.time()
  })


@app.route('/classify',methods=['POST'])
def classify():
  data = request.json
  if not data:
    return jsonify({'error':'No JSON body'}),400

  text = data.get('text','')
  if not text:
    return jsonify({'error':'text field required'}),400
  if len(text) < 10:
    return jsonify({'error':'text too short, minimum 10 characters'}),400

  if len(text) >10000:
    return jsonify({'error':'text too long, max 10000 chars'}),400

  encoding = tokenizer(
    text,
    max_length = 256,
    padding = 'max_length',
    truncation = True,
    return_tensors = 'pt'
  )

  input_ids = encoding['input_ids']
  attention_mask = encoding['attention_mask']

  with torch.no_grad():
    outputs = model(input_ids=input_ids,attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits,dim=1)
    confidence, predicted = probs.max(dim=1)
    category   = CATEGORIES[predicted.item()]
    confidence = confidence.item() * 100

    top3_probs, top3_idx = probs.topk(3, dim=1)
    top3 = [
        {"category": CATEGORIES[i], "confidence": f"{p*100:.1f}%"}
        for i, p in zip(top3_idx[0].tolist(), top3_probs[0].tolist())
    ]

    return jsonify({
        'status':     'success',
        'category':   category,
        'confidence': f"{confidence:.1f}%",
        'top3':       top3,
        'needs_review': confidence < 50
    })
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'internal server error'}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False)


from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json

from .models import EmotionClassifier

emotion_classifier = EmotionClassifier()


@csrf_exempt
@require_POST
def predict_emotion(request):
    try:
        data = json.loads(request.body.decode('utf-8'))
        sentence = data.get('sentence', '')
        print("sentence input => ", sentence)
        result = emotion_classifier.predict(sentence)
        return JsonResponse({'emotion': result})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

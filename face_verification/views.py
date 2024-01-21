# face_verification/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from .utils import verify_faces

@csrf_exempt
@require_POST
def face_verification_api(request):
    try:
        print("iwashere")
        # Get images from POST request
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')

        # Verify faces
        print("result")

        result = verify_faces(image1, image2)
        print("result")
        return JsonResponse({'result': result})

    except Exception as e:
        print (e)
        return JsonResponse({'error': str(e)})

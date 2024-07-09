from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Site
from .serializers import SiteSerializer

@api_view(['GET'])
def getData(request):
    # // example object
    # person = {'person':'aryan', 'age':22}
    if request.method == 'GET':
        output = [{"company_name": output.company_name, "url": output.url, "output": output.output}
                  for output in Site.objects.all()]
        return Response(output)


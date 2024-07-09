from rest_framework import serializers
from base.models import Site

#to serialise a python dictionary for a particular model(databse) into a json format.
class SiteSerializer(serailizers.ModelSerializer):
    class Meta:
        model = Site
        fields = ['id', 'company_name', 'url', 'output']
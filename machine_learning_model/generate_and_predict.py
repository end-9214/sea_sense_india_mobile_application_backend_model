import os
import sys
import django
import time
from datetime import datetime
import random
import pandas as pd
from predict import predict_activity_level  

# Set up Django environment
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.backend.settings")
django.setup()

from backend.seasense_backend.models import BeachPrediction


# List of beaches
beaches = [
    "Tithal Beach", "Dumas Beach", "Suvali Beach", "Umbharat Beach", "Dandi Beach", "Dabhari beach", "Diu Beach",
    "Mandavi Beach", "Khambhat Beach", "Aksa Beach", "Alibaug Beach", "Gorai Beach", "Juhu beach", "Manori Beach",
    "Marvé Beach", "Versova Beach", "Agardanda Beach", "Diveagar Beach", "Ganpatipule Beach", "Guhagar Beach",
    "Kelwa Beach", "Tarkarli Beach", "Shivaji Park Beach", "Anjarle Beach", "Dapoli Beach", "Dahanu Beach",
    "Srivardhan beach", "Kihim Beach", "Mandwa Beach", "Velneshwar Beach", "Vengurla Beach", "Bassein Beach",
    "Bhandarpule Beach", "Nagaon Beach", "Revdanda Beach", "Rewas Beach", "Kashid Beach", "Karde (Murud) Beach",
    "Harihareshwar Beach", "Bagmandla Beach", "Kelshi Beach", "Harnai Beach", "Bordi Beach", "Ratnagiri Beach",
    "Awas Beach", "Sasawne Beach", "Malvan Beach", "Agonda Beach", "Arambol Beach", "Benaulim Beach",
    "Cavelossim Beach", "Chapora Beach", "Mandrem Beach", "Palolem Beach", "Varca Beach", "Baga Beach",
    "Candolim Beach", "Calangute Beach", "Colva Beach", "Miramar Beach", "Morjim Beach", "Bambolim Beach",
    "Cabo de rama Beach", "Anjuna Beach", "Utorda Beach", "Majorda Beach", "Betalbatim Beach", "Sernabatim Beach",
    "Cavelossim Beach", "Mobor Beach", "Betul Beach", "Querim Beach", "Kalacha Beach", "Mandrem Beach",
    "Ashvem Beach", "Vagator Beach", "Ozran Beach", "Sinquerim Beach", "Coco Beach", "Kegdole Beach",
    "Caranzalem Beach", "Dona Paula Beach", "Vaiguinim Beach", "Siridao Beach", "Bogmalo Beach", "Baina Beach",
    "Hansa Beach", "Hollant Beach", "Cansaulim Beach", "Velsao Beach", "Canaiguinim Beach", "Kakolem Beach",
    "Dharvalem Beach", "Cola Beach", "Agonda Beach", "Palolem Beach", "Patnem Beach", "Rajbag Beach",
    "Talpona Beach", "Galgibag Beach", "Polem Beach", "Pebble Beach", "Karwar Beach", "Kudle beach",
    "Panambur Beach", "NITK Beach", "Sasihithlu Beach", "Maravanthe Beach", "Tannirubhavi Beach", "Malpe Beach",
    "Murudeshwara Beach", "Apsarakonda Beach", "Om Beach", "Kaup Beach", "Kodi Beach", "Someshwar Beach",
    "St Mary's Island Beach", "Mukka Beach", "Ullal beach", "Cherai Beach", "Chavakkad Beach", "Cherai Beach",
    "Fort Kochi beach", "Kollam Beach", "Kanhangad Beach", "Marari beach", "Meenkunnu Beach", "Muzhappilangad Beach",
    "Payyambalam Beach", "Saddam Beach", "Shangumughom Beach", "Snehatheeram Beach", "Kappil Beach Varkala",
    "Thirumullavaram Beach", "Kovalam Beach", "Hawa Beach", "Samudra Beach", "Lighthouse Beach", "Kannur Beach",
    "Kappad Beach", "Varkala Beach", "Padinjarekkara Beach", "Tanur Beach", "Azheekal Beach", "Alappuzha Beach",
    "Kozhikode Beach", "Bekal Beach", "Thiruvambadi Beach", "Kappil Beach", "Henry Island Beach", "Bakkhali sea beach",
    "Frasergunj Sea Beach", "Gangasagar Sea Beach", "Junput beach", "Bankiput Sea Beach", "Mandarmani beach",
    "Shankarpur Beach", "Tajpur beach", "Digha Sea Beach", "Udaypur Sea Beach", "Talsari Beach", "Dagara beach",
    "Chandipur-on-sea", "Gahirmatha Beach", "Satabhaya beach", "Pentha Sea Beach", "Hukitola beach",
    "Paradeep sea beach", "Astaranga beach", "Beleswar beach", "Konark Beach", "Chandrabhaga beach",
    "Ramachandi beach", "Puri Beach", "Satpada beach", "Parikud beach", "Ganjam beach", "Aryapalli beach",
    "Gopalpur-on-Sea", "Dhabaleshwar beach", "Ramayapatnam Beach", "Sonapur beach", "Sonpur Beach", "Donkuru Beach",
    "Nelavanka Beach", "Kaviti Beach", "Onturu Beach", "Ramayyapatnam Beach", "Baruva Beach", "Battigalluru Beach",
    "Sirmamidi Beach", "Ratti Beach", "Shivasagar Beach", "Dokulapadu Beach", "Nuvvalarevu Beach", "KR Peta Beach",
    "Bavanapadu Beach", "Mula Peta Beach", "BVS Beach", "Patha Meghavaram Beach", "Guppidipeta Beach",
    "Kotharevu Beach", "Rajaram Puram Beach", "Kalingapatnam Beach", "Bandaruvanipeta Beach", "Mogadalapadu Beach",
    "Vatsavalasa Beach", "S. Matchelesam Beach", "Balarampuram Beach", "Kunduvanipeta Beach", "PD Palem Beach",
    "Budagatlapalem Beach", "Kotcherla Beach", "Jeerupalem Beach", "Kovvada Beach", "Pothayyapeta Beach",
    "Chintapalli NGF Beach", "Chintapalli Beach", "Tammayyapalem Beach", "Konada Beach", "Divis Beach",
    "Bheemili Beach", "Mangamaripeta Beach", "Thotlakonda Beach", "Rushikonda Beach", "Sagarnagar Beach",
    "Jodugullapalem Beach", "RK Beach", "Durga Beach", "Yarada Beach", "Gagavaram Beach", "Adi's Beach",
    "Appikonda Beach", "Tikkavanipalem Beach", "Mutyalammapalem Beach", "Thanthadi Beach", "Seethapalem Beach",
    "Rambilli Beach", "Kothapatnam Beach", "Revupolavaram Beach", "Gudivada Beach", "Gurrajupeta Beach",
    "Pedhatheenarla Beach", "Rajjyapeta Beach", "Boyapadu Beach", "DLPuram Beach", "Pentakota Beach", "Rajavaram Beach",
    "Addaripeta Beach", "Danvaipeta Beach", "Gaddipeta Beach", "K. Perumallapuram Beach", "Konapapapeta Beach",
    "Uppada Beach", "Nemam Beach", "NTR Beach", "Seahorse Beach", "Dragonmouth Beach", "Pallam Beach",
    "Sunrise Beach", "Surasani Yanam Beach", "Vasalatippa Beach", "Odalarevu Beach", "Turpupalem Beach",
    "Kesanapalli Beach", "Shankaraguptham Beach", "Chintalamori Beach", "Natural Beach", "KDP Beach",
    "Antervedi Beach", "Pedamainavanilanka Beach", "Perupalem Beach", "Kanakadurga Beach", "Gollapalem Beach",
    "Podu Beach", "Gollapalem Beach", "Pedapatnam Beach", "Modi Beach", "Tallapalem Beach", "Manginapudi Beach",
    "Crab Beach", "Gopuvanipalem Beach", "Lonely Beach", "Chinakaragraharam Beach", "Destiny Beach",
    "Machilipatnam Beach", "Hamsaladeevi Beach", "Diviseema Beach", "Dindi Beach", "Nizampatnam Beach",
    "Suryalanka Beach", "Pandurangapuram Beach", "Vodarevu Beach", "Ramachandrapuram Beach", "Motupalli Beach",
    "Chinaganjam Beach", "Pedaganjam Beach", "Kanapurthi Beach", "Kodurivaripalem Beach", "Katamvaripalem Beach",
    "Kanuparthi Beach", "Motumala Beach", "Pinnivaripalem Beach", "Kothapatnam Beach", "Gavandlapallem Beach",
    "Rajupalem Beach", "Etthamukhala Beach", "Madanur Beach", "White sand Beach", "Pakka Beach", "Pakala Beach",
    "Ullapalem Beach", "Pedda Pallepalem Beach", "Karedu Beach", "G-Star Shiv Beach", "Shiv satendra Prajapati Beach",
    "Alagayapalem Beach", "Chackicherla Beach", "Ramayapattanam Public Beach", "Karla palem Beach", "SSR Port Beach",
    "Pallipalem Public Beach", "Kotha sathram Beach", "Pedaramudu palem Beach", "Chinnaramudu palem Beach",
    "Thummalapenta Beach", "Thatichetla Palem Beach", "LN Puram Beach", "Iskapalli Beach", "Ponnapudi Beach",
    "Ramathirdamu Beach", "Govundlapalem Beach", "Kudithipalem Beach", "Gangapatnam Beach", "Mypadu Beach",
    "Zard Beach", "Kotha Koduru Beach", "Koduru Beach", "Katepalli Beach", "Nelaturu Beach", "Krishnapatnam Beach",
    "Theegapalem Beach", "Srinivasa satram Beach", "Pattapupalem Beach", "Moonside Beach", "Thupilipalem Beach",
    "Kondurpalem Beach", "Alone Beach", "Raviguntapalem Beach", "Nawabpet Beach", "Marina Beach",
    "Edward Elliot's Beach", "Kasimedu's N4 Beach", "Golden Beach", "Thiruvanmayur Beach", "Silver Beach",
    "Covelong Beach", "Mahabalipuram Beach", "Olaikuda Beach", "Ariyaman/kushi Beach", "Pamban Beach",
    "Dhanushkodi Beach", "Velankanni Beach", "Sothavilai Beach", "Kanyakumari Beach", "Vattakotai Beach",
    "Sanguthurai Beach", "Sengumal Beach", "Thoothukudi Beach", "Tiruchendur Beach", "Poompuhar beach",
    "Promenade Beach", "Karaikal Beach", "Yanam Beach", "Auroville Beach", "Paradise Beach", "Serenity Beach"
]

# Function to generate random sample data for a beach
def generate_sample(beach, timestamp):
    sample = {
        'Date & Time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'Sea Surface Temp (°C)': round(random.uniform(25.0, 30.0), 1),
        'Air Temp (°C)': round(random.uniform(20.0, 35.0), 1),
        'Wind Speed (km/h)': round(random.uniform(0.0, 10.0), 1),
        'Wave Height (m)': round(random.uniform(0.0, 1.0), 2),
        'UV Index': round(random.uniform(0.0, 12.0), 1),
        'Hour': timestamp.hour,
        'dayOfweek': timestamp.weekday()
    }
    return sample

# Function to generate data every hour for each beach and save predictions to the database
def generate_and_predict_data(beaches, interval=3600):
    while True:
        timestamp = datetime.now()
        for beach in beaches:
            sample = generate_sample(beach, timestamp)
            predicted_activity_level = predict_activity_level(sample)

            # Save the prediction to the database
            BeachPrediction.objects.create(
                beach_name=beach,
                timestamp=timestamp,
                sea_surface_temp=sample['Sea Surface Temp (°C)'],
                air_temp=sample['Air Temp (°C)'],
                wind_speed=sample['Wind Speed (km/h)'],
                wave_height=sample['Wave Height (m)'],
                uv_index=sample['UV Index'],
                activity_level=predicted_activity_level
            )

        time.sleep(interval)

# Start generating and predicting data
generate_and_predict_data(beaches)
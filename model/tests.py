from django.test import TestCase, Client
from model.models import ClassificationResult
from .models import ClassificationResult
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
import os
import pickle
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from sklearn.tree import DecisionTreeClassifier
# Create your tests here.
class ClassificationResultModelTest(TestCase):
    def setUp(self):
        ClassificationResult.objects.create(
            varietas="Varietas A",
            warna="Merah",
            rasa="Manis",
            musim="Musim Panas",
            penyakit="Tidak Ada",
            teknik="Organik",
            ph=6.5,
            boron=0.3,
            fosfor=0.4,
            prediction="Prediksi A"
        )

    def test_classification_result_creation(self):
        classification_result = ClassificationResult.objects.get(varietas="Varietas A")
        
        self.assertEqual(classification_result.warna, "Merah")
        self.assertEqual(classification_result.rasa, "Manis")
        self.assertEqual(classification_result.musim, "Musim Panas")
        self.assertEqual(classification_result.penyakit, "Tidak Ada")
        self.assertEqual(classification_result.teknik, "Organik")
        self.assertEqual(classification_result.ph, 6.5)
        self.assertEqual(classification_result.boron, 0.3)
        self.assertEqual(classification_result.fosfor, 0.4)
        self.assertEqual(classification_result.prediction, "Prediksi A")

    def test_classification_result_str(self):
        classification_result = ClassificationResult.objects.get(varietas="Varietas A")
        
        self.assertEqual(str(classification_result), "Varietas A, Merah, Prediksi A")

class UserRegistrationTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = reverse('register')  # Asumsikan ada URL pattern dengan nama 'register'

    def test_register_get(self):
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'register.html')

    def test_register_post_valid(self):
        response = self.client.post(self.register_url, {
            'username': 'petanikode1',
            'password1': 'Passwordnya123*',
            'password2': 'Passwordnya123*',
        })
        self.assertEqual(response.status_code, 302)  # Redirection after successful registration
        self.assertRedirects(response, reverse('login'))
        self.assertTrue(User.objects.filter(username='petanikode1').exists())

    def test_register_post_invalid(self):
        response = self.client.post(self.register_url, {
            'username': 'testuser',
            'password1': 'complexpassword123',
            'password2': 'wrongpassword',
        })
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'register.html')
        self.assertFalse(User.objects.filter(username='testuser').exists())

class UserLoginTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.login_url = reverse('login')  # Asumsikan ada URL pattern dengan nama 'login'
        self.home_url = reverse('home')  # Asumsikan ada URL pattern dengan nama 'home'
        self.user = User.objects.create_user(username='petanikode1', password='Passwordnya123*')

    def test_login_get(self):
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'login.html')
        self.assertIsInstance(response.context['form'], AuthenticationForm)

    def test_login_post_valid(self):
        response = self.client.post(self.login_url, {
            'username': 'petanikode1',
            'password': 'Passwordnya123*',
        })
        self.assertEqual(response.status_code, 302)  # Redirection setelah sukses login
        self.assertRedirects(response, self.home_url)
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_login_post_invalid(self):
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'wrongpassword',
        })
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'login.html')
        self.assertFalse(response.context['form'].is_valid())
        self.assertContains(response, "Invalid username or password.")
        self.assertFalse(response.wsgi_request.user.is_authenticated)

class PredictionTestCase(TestCase):
    def setUp(self):
        # Set up a test client
        self.client = Client()

        # Create a sample model and save it to a pickle file
        x_train = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.67, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ])
        y_train = np.array([0, 1])
        model = DecisionTreeClassifier().fit(x_train, y_train)
        with open('model_m03.pkl', 'wb') as f:
            pickle.dump(model, f)

    def tearDown(self):
        # Remove the pickle file after the tests
        if os.path.exists('model_m03.pkl'):
            os.remove('model_m03.pkl')

    def test_prediction_view(self):
        # Test the prediction view
        response = self.client.post(reverse('prediction'), {
            'Varietas': '0.0',
            'Warna': '0.67',
            'rasa': '0.0',
            'Musim': '0.0',
            'Penyakit': '0.0',
            'teknik': '0.0',
            'PH': '0.0',
            'boron': '2.5',
            'fosfor': '6.5',
            'prediction': 'Predict'
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Kelas A')  # Assuming prediction is Kelas A for this input

        # Check if the result is saved in the database
        result = ClassificationResult.objects.first()
        self.assertIsNotNone(result)
        self.assertEqual(result.varietas, 'Beras Hitam')
        self.assertEqual(result.warna, 'Merah')
        self.assertEqual(result.rasa, 'Pulen')
        self.assertEqual(result.musim, 'Hujan')
        self.assertEqual(result.penyakit, 'Burung')
        self.assertEqual(result.teknik, 'Jajar Legowo')
        self.assertEqual(result.ph, '2')
        self.assertEqual(result.boron, 2.5)
        self.assertEqual(result.fosfor, 6.5)
        self.assertEqual(result.prediction, 'Kelas A')  # Assuming prediction is Kelas A for this input

import unittest
import os
import sys
import tempfile
from flask import Flask
from flask.testing import FlaskClient
import json

# Add your app directory to path
sys.path.append('E:/bone_fracture_system')

from app import app, users, reports, detect_fracture

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and test data"""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()
        
        # Create test image
        self.test_image_path = self.create_test_image()
        
        # Backup original data
        self.original_users = users.copy()
        self.original_reports = reports.copy()
    
    def tearDown(self):
        """Restore original data"""
        users.clear()
        users.update(self.original_users)
        reports.clear()
        reports.extend(self.original_reports)
        
        # Clean up test files
        if os.path.exists(self.test_image_path):
            os.path.exists(self.test_image_path)
    
    def create_test_image(self):
        """Create a test X-ray image"""
        import cv2
        import numpy as np
        
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (150, 150), (250, 250), (100, 100, 100), -1)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_file.name, img)
        return temp_file.name
    
    def test_user_authentication_flow(self):
        """Test complete user authentication flow"""
        # Test login with valid credentials
        response = self.client.post('/login', data={
            'user_type': 'patient',
            'email': 'patient@test.com',
            'password': '123'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Patient Portal', response.data)
    
    def test_patient_dashboard_access(self):
        """Test patient dashboard access after login"""
        with self.client:
            # Login first
            self.client.post('/login', data={
                'user_type': 'patient',
                'email': 'patient@test.com',
                'password': '123'
            })
            
            # Access dashboard
            response = self.client.get('/patient_dashboard')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Medical Overview', response.data)
    
    def test_doctor_dashboard_access(self):
        """Test doctor dashboard access after login"""
        with self.client:
            # Login as doctor
            self.client.post('/login', data={
                'user_type': 'doctor',
                'email': 'doctor@test.com',
                'password': '123'
            })
            
            # Access dashboard
            response = self.client.get('/doctor_dashboard')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Practice Overview', response.data)
    
    def test_report_creation_flow(self):
        """Test complete report creation flow"""
        with self.client:
            # Login as patient
            self.client.post('/login', data={
                'user_type': 'patient',
                'email': 'patient@test.com',
                'password': '123'
            })
            
            # Upload X-ray and create report
            with open(self.test_image_path, 'rb') as img:
                response = self.client.post('/upload_xray', data={
                    'xray_image': img,
                    'doctor_id': 1
                }, follow_redirects=True)
            
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Analysis Complete', response.data)
    
    def test_report_viewing_flow(self):
        """Test report viewing functionality"""
        with self.client:
            # Login
            self.client.post('/login', data={
                'user_type': 'patient',
                'email': 'patient@test.com',
                'password': '123'
            })
            
            # View reports
            response = self.client.get('/view_reports')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Medical History', response.data)
    
    def test_user_registration_flow(self):
        """Test new user registration"""
        response = self.client.post('/signup', data={
            'user_type': 'patient',
            'name': 'Test User',
            'email': 'test@example.com',
            'password': 'test123'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        # Check if user was added
        new_user = next((u for u in users['patients'] if u['email'] == 'test@example.com'), None)
        self.assertIsNotNone(new_user)
    
    def test_fracture_detection_integration(self):
        """Test fracture detection integrated with Flask app"""
        # Test with your actual detection function
        result = detect_fracture(self.test_image_path)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        required_keys = ['fracture_type', 'severity', 'confidence', 'bounding_boxes', 'processed_image', 'fracture_detected']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_session_management(self):
        """Test user session management"""
        with self.client:
            # Login
            response = self.client.post('/login', data={
                'user_type': 'patient',
                'email': 'patient@test.com',
                'password': '123'
            })
            
            # Check if session is set
            with self.client.session_transaction() as session:
                self.assertTrue(session['loggedin'])
                self.assertEqual(session['user_type'], 'patient')
                self.assertEqual(session['email'], 'patient@test.com')
            
            # Test logout
            response = self.client.get('/logout', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'MedScan AI', response.data)

if __name__ == '__main__':
    unittest.main(verbosity=2)
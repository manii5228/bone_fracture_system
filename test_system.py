import unittest
import os
import sys
import tempfile
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

class TestSystem(unittest.TestCase):
    
    def setUp(self):
        """Set up Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.base_url = "http://localhost:8000"
        self.wait = WebDriverWait(self.driver, 10)
        
        # Create test image
        self.test_image_path = self.create_test_image()
    
    def tearDown(self):
        """Clean up after tests"""
        self.driver.quit()
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def create_test_image(self):
        """Create a test X-ray image for upload"""
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        img = Image.new('RGB', (400, 400), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 300, 300], fill='gray')
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img.save(temp_file.name)
        return temp_file.name
    
    def test_complete_user_journey(self):
        """Test complete user journey from login to report viewing"""
        try:
            # 1. Navigate to homepage
            self.driver.get(self.base_url)
            self.assertIn("MedScan AI", self.driver.title)
            
            # 2. Go to login page
            login_link = self.wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Get Started"))
            )
            login_link.click()
            
            # 3. Perform login
            self.wait.until(EC.presence_of_element_located((By.NAME, "email")))
            
            user_type = self.driver.find_element(By.NAME, "user_type")
            email = self.driver.find_element(By.NAME, "email")
            password = self.driver.find_element(By.NAME, "password")
            
            # Select patient type
            from selenium.webdriver.support.ui import Select
            select = Select(user_type)
            select.select_by_value("patient")
            
            # Enter credentials
            email.send_keys("patient@test.com")
            password.send_keys("123")
            
            # Submit login
            submit_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Sign In')]")
            submit_btn.click()
            
            # 4. Verify dashboard access
            self.wait.until(EC.title_contains("Patient Dashboard"))
            self.assertIn("Patient Portal", self.driver.page_source)
            
            # 5. Navigate to upload page
            upload_link = self.wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Upload X-Ray"))
            )
            upload_link.click()
            
            # 6. Verify upload page
            self.wait.until(EC.title_contains("Upload X-Ray"))
            self.assertIn("Medical Image Analysis", self.driver.page_source)
            
            print("✅ System test completed successfully!")
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            raise
    
    def test_doctor_workflow(self):
        """Test doctor-specific workflow"""
        try:
            # 1. Navigate to login
            self.driver.get(f"{self.base_url}/login")
            
            # 2. Login as doctor
            self.wait.until(EC.presence_of_element_located((By.NAME, "email")))
            
            user_type = self.driver.find_element(By.NAME, "user_type")
            email = self.driver.find_element(By.NAME, "email")
            password = self.driver.find_element(By.NAME, "password")
            
            from selenium.webdriver.support.ui import Select
            select = Select(user_type)
            select.select_by_value("doctor")
            
            email.send_keys("doctor@test.com")
            password.send_keys("123")
            
            submit_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Sign In')]")
            submit_btn.click()
            
            # 3. Verify doctor dashboard
            self.wait.until(EC.title_contains("Medical Dashboard"))
            self.assertIn("Medical Professional Portal", self.driver.page_source)
            
            # 4. Check patient reports access
            reports_link = self.wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Patient Reports"))
            )
            reports_link.click()
            
            # 5. Verify reports page
            self.wait.until(EC.title_contains("Patient Reports"))
            
            print("✅ Doctor workflow test completed successfully!")
            
        except Exception as e:
            print(f"❌ Doctor workflow test failed: {e}")
            raise
    
    def test_responsive_design(self):
        """Test responsive design on different screen sizes"""
        try:
            # Test mobile view
            self.driver.set_window_size(375, 667)  # iPhone 6/7/8
            self.driver.get(self.base_url)
            
            # Check if page loads without horizontal scroll
            body_width = self.driver.execute_script("return document.body.scrollWidth")
            viewport_width = self.driver.execute_script("return window.innerWidth")
            
            self.assertLessEqual(body_width, viewport_width, 
                               "Page should be responsive on mobile")
            
            # Test tablet view
            self.driver.set_window_size(768, 1024)  # iPad
            self.driver.get(self.base_url)
            
            body_width = self.driver.execute_script("return document.body.scrollWidth")
            viewport_width = self.driver.execute_script("return window.innerWidth")
            
            self.assertLessEqual(body_width, viewport_width,
                               "Page should be responsive on tablet")
            
            print("✅ Responsive design test completed successfully!")
            
        except Exception as e:
            print(f"❌ Responsive design test failed: {e}")
            raise
    
    def test_performance(self):
        """Test application performance"""
        try:
            start_time = time.time()
            
            # Navigate through main pages
            self.driver.get(self.base_url)
            load_time_home = time.time() - start_time
            
            self.driver.get(f"{self.base_url}/login")
            load_time_login = time.time() - start_time - load_time_home
            
            # Performance thresholds (in seconds)
            self.assertLess(load_time_home, 3, "Homepage should load within 3 seconds")
            self.assertLess(load_time_login, 2, "Login page should load within 2 seconds")
            
            print(f"✅ Performance test completed: Homepage={load_time_home:.2f}s, Login={load_time_login:.2f}s")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            raise

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print detailed results
    print(f"\n{'='*50}")
    print("SYSTEM TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
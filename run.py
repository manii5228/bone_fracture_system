from app import app

if __name__ == '__main__':
    print("🚀 Starting Bone Fracture Detection System...")
    print("📡 Server running at: http://localhost:8000")
    print("🦴 AI-Powered Fracture Detection Ready!")
    print("⏹️  Press CTRL+C to stop the server")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=8000)
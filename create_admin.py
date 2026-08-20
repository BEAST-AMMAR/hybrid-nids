import sys
import os

from src.backend.database import SessionLocal, engine
from src.backend import models, auth

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()

admin_email = "admin@admin.com"
try:
    user = db.query(models.User).filter(models.User.email == admin_email).first()
    if not user:
        hashed_password = auth.get_password_hash("admin")
        admin_user = models.User(email=admin_email, hashed_password=hashed_password, full_name="Admin", is_active=True)
        db.add(admin_user)
        db.commit()
        print(f"Admin user created successfully! Email: {admin_email}, Password: admin")
    else:
        print(f"Admin user already exists with Email: {admin_email}")
        
    users = db.query(models.User).all()
    print("All users:", [(u.email, u.full_name) for u in users])
except Exception as e:
    print("Error:", e)
finally:
    db.close()

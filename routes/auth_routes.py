from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from database import users_collection, centers_collection
from auth import get_password_hash, verify_password, create_access_token, get_current_user
from bson import ObjectId

router = APIRouter()

class RegisterRequest(BaseModel):
    center_name: str
    email: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register")
async def register(request: RegisterRequest):
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": request.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create Center first
    # In a real SaaS, check if center name already exists or handle accordingly
    existing_center = await centers_collection.find_one({"name": request.center_name})
    if existing_center:
        raise HTTPException(status_code=400, detail="Center name already registered")
        
    center = {
        "name": request.center_name,
        "subscription_tier": "free",
        "is_active": True
    }
    new_center = await centers_collection.insert_one(center)
    center_id = new_center.inserted_id

    # Create User
    hashed_password = get_password_hash(request.password)
    user = {
        "email": request.email,
        "full_name": request.full_name,
        "hashed_password": hashed_password,
        "center_id": center_id,
        "role": "admin"
    }
    await users_collection.insert_one(user)

    return {"message": "Registration successful. Please login."}

@router.post("/login")
async def login(request: LoginRequest):
    user = await users_collection.find_one({"email": request.email})
    if not user or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": str(user["_id"])}
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "center_id": str(user["center_id"])
        }
    }

@router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "center_id": current_user["center_id"]
    }

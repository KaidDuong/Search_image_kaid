# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy.sql import func
from sqlalchemy import  Column, Integer, String, Float, DateTime, ForeignKey

from app import db
from sqlalchemy.orm import relationship
    
class Document(db.Model):
    __tablename__ = 'Documents'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    total_page = Column(Integer)
    annotations = Column(String)
    createdOn = Column(DateTime(timezone=True), default=func.now())
    createdBy = Column(String)

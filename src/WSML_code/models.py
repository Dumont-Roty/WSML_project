from pydantic import BaseModel, Field
from typing import List, Optional

class Movie(BaseModel):
    url: str
    title: str
    year: int
    directors: List[str] = Field(default_factory=list)
    casting: List[str] = Field(default_factory=list)
    duration: Optional[int] = None
    nbr_watched: Optional[int] = None
    nbr_appearence: Optional[int] = None
    nbr_likes: Optional[int] = None
    rating: Optional[float] = None
    fans_favoris: Optional[int] = None
    producers: List[str] = Field(default_factory=list)
    writers: List[str] = Field(default_factory=list)
    composer: Optional[List[str]] = Field(default_factory=list)
    studio: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    budget: Optional[int] = None
    revenue: Optional[int] = None
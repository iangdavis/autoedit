#!/usr/bin/env python3
"""
autoedit local API server.

Endpoints:
  GET /instagram/{handle}   — fetch last N post image URLs from a public profile
  GET /image?url=...        — proxy an image (avoids CORS on Instagram CDN URLs)

Usage:
  pip install fastapi uvicorn httpx
  uvicorn server:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import httpx

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

IG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.instagram.com/",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
}


@app.get("/instagram/{handle}")
async def get_instagram_posts(handle: str, count: int = 12):
    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={handle}"
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers={**IG_HEADERS, "x-ig-app-id": "936619743392459", "x-requested-with": "XMLHttpRequest"})
        except httpx.RequestError as e:
            raise HTTPException(502, f"Could not reach Instagram: {e}")

    if resp.status_code == 404:
        raise HTTPException(404, "Profile not found — check the handle")
    if resp.status_code == 401:
        raise HTTPException(401, "Profile is private")
    if not resp.is_success:
        raise HTTPException(resp.status_code, f"Instagram returned {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(502, f"status={resp.status_code} body={resp.text[:400]!r}")

    user = data.get("data", {}).get("user") or data.get("graphql", {}).get("user", {})
    edges = user.get("edge_owner_to_timeline_media", {}).get("edges", [])
    if not edges:
        raise HTTPException(404, "No posts found — profile may be private")

    image_urls = []
    for edge in edges[:count]:
        node = edge["node"]
        carousel = node.get("edge_sidecar_to_children", {}).get("edges", [])
        if carousel:
            url = carousel[0]["node"].get("display_url")
        else:
            url = node.get("display_url") or node.get("thumbnail_src")
        if url:
            image_urls.append(url)

    return {"handle": handle, "count": len(image_urls), "urls": image_urls}


@app.get("/image")
async def proxy_image(url: str = Query(...)):
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": IG_HEADERS["User-Agent"]})
        except httpx.RequestError as e:
            raise HTTPException(502, str(e))
    if not resp.is_success:
        raise HTTPException(resp.status_code, "Image fetch failed")
    content_type = resp.headers.get("content-type", "image/jpeg")
    return Response(content=resp.content, media_type=content_type)

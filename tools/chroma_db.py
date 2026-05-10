"""
ChromaDB Utility for Market Data Persistence
-------------------------------------------
Handles saving technical analysis points for future LLM processing.
"""

import os
import time
from datetime import datetime
import chromadb
from chromadb.config import Settings

class MarketDB:
    def __init__(self, path="./chroma_db"):
        """Initialize ChromaDB client with local persistence."""
        # Ensure path exists
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        self.client = chromadb.PersistentClient(path=path)
        # We use a single collection for all technical market data
        self.collection = self.client.get_or_create_collection(
            name="market_data",
            metadata={"hnsw:space": "cosine"}
        )

    def save_point(self, symbol, point_type, price, strength=1.0, metadata=None):
        """
        Save a generic market point to the database.
        
        Args:
            symbol:     Trading symbol (e.g., XAUUSDm)
            point_type: Type of point (support, resistance, liquidity_zone, reversal)
            price:      Price level
            strength:   Score or strength of the point (0-1)
            metadata:   Additional context
        """
        timestamp = time.time()
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Unique ID combining symbol, type, and high-res timestamp
        doc_id = f"{symbol}_{point_type}_{int(timestamp * 1000)}"
        
        # Metadata for filtering
        meta = {
            "symbol": symbol,
            "type": point_type,
            "price": float(price),
            "strength": float(strength),
            "timestamp": timestamp,
            "date": date_str
        }
        if metadata:
            meta.update(metadata)
            
        # Text representation for LLM/RAG
        text = f"Market Alert | {date_str} | {symbol} detected {point_type} at {price:.5f} with strength {strength:.2f}."
        if metadata:
            text += f" Context: {metadata}"
        
        try:
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta]
            )
            # print(f"💾 Saved {point_type} to ChromaDB: {price:.5f}")
        except Exception as e:
            print(f"❌ Error saving to ChromaDB: {e}")

    def save_sr_levels(self, symbol, supports, resistances):
        """Batch save S/R levels."""
        for s in supports:
            self.save_point(symbol, "support", s)
        for r in resistances:
            self.save_point(symbol, "resistance", r)

    def save_liquidity_zones(self, symbol, zones):
        """Batch save liquidity zones from liquidity_zones.py."""
        for z in zones:
            self.save_point(
                symbol, 
                "liquidity_zone", 
                z["level"], 
                z["strength"], 
                {"zone_type": z["type"], "bar_ago": z["bar_ago"]}
            )

    def save_reversal(self, symbol, price, prediction_val):
        """Save a confirmed trend reversal point."""
        direction = "bullish" if prediction_val > 0 else "bearish"
        self.save_point(
            symbol, 
            "reversal", 
            price, 
            strength=abs(prediction_val),
            metadata={"direction": direction, "prediction": prediction_val}
        )

# Create a singleton instance for the app
db = MarketDB()

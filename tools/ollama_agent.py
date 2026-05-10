"""
Ollama Trading Agent
--------------------
Uses llama3.2 to determine trading actions based on market context.
"""

import ollama
import json
import re

class OllamaAgent:
    def __init__(self, model="llama3.2:latest"):
        self.model = model
        self.system_prompt = """
You are an expert Financial Trading AI (Llama 3.2). Your goal is to maximize profit through active adaptation.
You will be provided with a Multi-Timeframe Market Snapshot (M1, M5, H1) and RECENT PERFORMANCE LOGS.

[AVAILABLE ACTIONS]
- BUY: Open a new buy position.
- SELL: Open a new sell position.
- CLOSE_ALL: Close all open positions immediately.
- HOLD: Do nothing.

[STRATEGY RULES]
1. ACTIVE LEARNING & CORRECTION: 
   - Analyze [RECENT LOGS]. If you see a series of losses in one direction (e.g., BUYing), you are likely fighting a trend.
   - Do NOT just HOLD. Instead, CORRECT YOUR BIAS. If buying failed repeatedly, look for the next logical SELL setup.
   - Use the losses to identify "False Breakouts" and trade the reversal.
2. ALIGNMENT & MOMENTUM:
   - Primary: M1 and M5 must align for entry. 
   - Secondary: H1 provides the macro bias. If M1+M5 align AGAINST H1, it's a "Counter-Trend Scalp" (take it, but be quick).
3. CORRELATION EDGE: Use Gold/USDCHF or BTC/Nasdaq correlations as a "Confirmation Trigger." 
4. TECHNICAL AGGRESSION: Buy at confirmed support, Sell at confirmed resistance. If price breaks a level, trade the momentum.
5. NO INHIBITION: Do not stop trading because of losses. Use the data to trade SMARTER, not LESS.

[OUTPUT FORMAT]
Your response must contain a JSON block followed by a brief reasoning.
{
  "action": "BUY/SELL/HOLD/CLOSE_ALL",
  "reasoning": "Identify the mistake in recent logs (if any) and explain how this trade corrects it while aligning with current M1/M5 momentum."
}
"""

    def get_decision(self, market_context_text):
        """
        Send context to Ollama and return the parsed decision.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this snapshot and decide:\n\n{market_context_text}"}
                ]
            )
            
            raw_content = response['message']['content']
            
            # Extract JSON block
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                return decision
            else:
                # Fallback if LLM doesn't output JSON
                return {"action": "HOLD", "reasoning": "Failed to parse LLM response: " + raw_content[:100]}
                
        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            return {"action": "HOLD", "reasoning": "Ollama exception occurred."}

# Singleton
agent = OllamaAgent()

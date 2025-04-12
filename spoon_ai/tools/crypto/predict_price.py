from .base import BitqueryTool
from datetime import datetime, timedelta

class PredictPrice(BitqueryTool):
    name: str = "predict_price"
    description: str = "Predicts the price of a token"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_address": {
                "type": "string",
                "description": "The address of the token"
            }
        }
    }
    graph_template: str = """
query ($baseAddress: String, $interval: Int) {{
  ethereum(network: ethereum) {{
    dexTrades(
      baseCurrency: {{is: $baseAddress}}
      date: {{since: "{since}", till: "{till}"}}
      options: {{limit: 1000, desc: "timeInterval.minute"}}
      quoteCurrency: {{is: "{token_address}"}}
      priceAsymmetry: {{lt: 1}}
    ) {{
      timeInterval {{
        minute(count: $interval)
      }}
      sellCurrency: quoteCurrency {{
        address
      }}
      avg: quotePrice(calculate: average)
    }}
  }}
}}
"""
    async def execute(self, token_address: str) -> str:
        since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        till = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        response = await super().execute(token_address=token_address, since=since, till=till)
        
        trade_data = response['data']['ethereum']['dexTrades']
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
        df = pd.json_normalize(trade_data)
        df['timeInterval.minute'] = pd.to_datetime(df['timeInterval.minute'])
        df['timeInterval.minute'] = df['timeInterval.minute'].astype(int) / 10**9
        features = ['timeInterval.minute']
        target = ['avg']
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100)
        
        model.fit(X_train, y_train)
        
        predict_since = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        predict_till = datetime.now().strftime("%Y-%m-%d")
        
        predict_response = await super().execute(token_address=token_address, since=predict_since, till=predict_till)
        predict_trade_data = predict_response['data']['ethereum']['dexTrades']
        
        predict_df = pd.json_normalize(predict_trade_data)
        predict_df['timeInterval.minute'] = pd.to_datetime(predict_df['timeInterval.minute'])
        predict_df['timeInterval.minute'] = predict_df['timeInterval.minute'].astype(int) / 10**9
        predict_X = predict_df[features]
        predict_X = scaler.transform(predict_X)
        
        predict_y = model.predict(predict_X)
        
        predict_df['predicted_price'] = predict_y
        
        return predict_df.to_json(orient='records')
        
        
        
        
        
        
        
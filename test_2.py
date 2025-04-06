import os
import asyncio
import json
import sys

# Add parent directory to path if LST tool is not in Python path
# sys.path.append('/path/to/parent/directory')

from spoon_ai.tools.dex.lst_arbitrage import LstArbitrageTool

async def run_basic_test():
    """Run basic tests to check if the tool works correctly"""
    print("Initializing LST Arbitrage Tool...")
    
    # Create tool instance
    lst_tool = LstArbitrageTool()
    
    # Simple test - Get current gas price
    try:
        print("\n[Test 1] Get current gas price")
        gas_price = await lst_tool.get_current_gas_price()
        print(f"Current gas price: {gas_price} Gwei")
        print("✓ Gas price fetched successfully")
    except Exception as e:
        print(f"✗ Failed to fetch gas price: {str(e)}")
        return
    
    # Test fetching token exchange data
    try:
        print("\n[Test 2] Get ETH/stETH exchange data")
        exchange_data = await lst_tool.get_token_exchange_data("ETH", "stETH")
        print(f"ETH/stETH exchange rate: {exchange_data['price']}")
        print(f"Exchange: {exchange_data.get('exchange', 'Unknown')}")
        print("✓ Exchange data fetched successfully")
    except Exception as e:
        print(f"✗ Failed to fetch exchange data: {str(e)}")
        return
    
    # Simple analysis of ETH to stETH arbitrage
    try:
        print("\n[Test 3] Analyze ETH to stETH arbitrage")
        result = await lst_tool.execute(from_token="ETH", to_token="stETH", amount=1.0)
        
        if "arbitrage_analysis" in result and "best_path" in result["arbitrage_analysis"]:
            best_path = result["arbitrage_analysis"]["best_path"]
            print(f"Best path: {best_path.get('route_type', 'unknown')}")
            print(f"Input: 1.0 ETH, Output: {best_path.get('output_amount', 0):.6f} stETH")
            if "profit_percent" in best_path:
                print(f"Profit margin: {best_path['profit_percent']:.2f}%")
            print("✓ Arbitrage analysis successful")
        else:
            print("✗ Could not find arbitrage path")
            if "arbitrage_analysis" in result and "error" in result["arbitrage_analysis"]:
                print(f"Error: {result['arbitrage_analysis']['error']}")
    except Exception as e:
        print(f"✗ Arbitrage analysis failed: {str(e)}")

async def run_comprehensive_test():
    """Run more comprehensive tests, analyze various arbitrage scenarios"""
    print("\n===== Comprehensive Test of LST Arbitrage Tool =====")
    
    # Create tool instance with custom configuration
    config = {
        "risk_preference": 3.0,  # Lower risk preference to find arbitrage opportunities more easily
        "min_profit_threshold": 0.00001  # Lower minimum profit threshold for testing
    }
    
    lst_tool = LstArbitrageTool(config=config)
    
    # Define list of tokens to test
    test_tokens = ["ETH", "stETH", "rETH", "cbETH", "wstETH"]
    
    # Test arbitrage opportunities under different risk preferences
    try:
        print("\n[Test 4] Arbitrage analysis with different risk preferences")
        risk_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        for risk in risk_levels:
            print(f"\nRisk preference: {risk}")
            lst_tool.update_config({"risk_preference": risk})
            result = await lst_tool.find_circular_arbitrage("ETH", 1.0)
            if "best_opportunity" in result:
                opp = result["best_opportunity"]
                print(f"Profit: {opp['profit']:.6f} ETH ({opp['profit_percent']:.2f}%)")
            else:
                print("No arbitrage opportunity found")
    except Exception as e:
        print(f"✗ Risk preference test failed: {str(e)}")
    
    # Test arbitrage with different amount sizes
    try:
        print("\n[Test 5] Arbitrage analysis with different amount sizes")
        amounts = [0.1, 1.0, 5.0, 10.0, 50.0]
        for amount in amounts:
            print(f"\nTest amount: {amount} ETH")
            result = await lst_tool.find_circular_arbitrage("ETH", amount)
            if "best_opportunity" in result:
                opp = result["best_opportunity"]
                print(f"Profit: {opp['profit']:.6f} ETH ({opp['profit_percent']:.2f}%)")
                print(f"Gas cost: {opp['gas_cost_eth']:.6f} ETH")
            else:
                print("No arbitrage opportunity found")
    except Exception as e:
        print(f"✗ Amount size test failed: {str(e)}")
    
    # Test arbitrage opportunities between all token pairs
    try:
        print("\n[Test 6] Comprehensive token pair arbitrage analysis")
        for from_token in test_tokens:
            for to_token in test_tokens:
                if from_token != to_token:
                    print(f"\nAnalyzing {from_token} → {to_token}")
                    result = await lst_tool.analyze_arbitrage_path(from_token, to_token, 1.0)
                    
                    if "best_path" in result:
                        best_path = result["best_path"]
                        print(f"Best path: {best_path.get('route_type', 'unknown')}")
                        print(f"Input: 1.0 {from_token}, Output: {best_path.get('output_amount', 0):.6f} {to_token}")
                        print(f"Profit margin: {best_path.get('profit_percent', 0):.2f}%")
                        print(f"Gas cost: {best_path.get('gas_cost_eth', 0):.6f} ETH")
                    else:
                        print(f"Could not find arbitrage path")
    except Exception as e:
        print(f"✗ Token pair analysis failed: {str(e)}")
    
    # Test slippage impact
    try:
        print("\n[Test 7] Slippage impact analysis")
        slippage_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1% to 5%
        for slippage in slippage_levels:
            print(f"\nTest slippage: {slippage*100}%")
            lst_tool.update_config({"slippage_tolerance": slippage})
            result = await lst_tool.find_circular_arbitrage("ETH", 1.0)
            if "best_opportunity" in result:
                opp = result["best_opportunity"]
                print(f"Profit: {opp['profit']:.6f} ETH ({opp['profit_percent']:.2f}%)")
            else:
                print("No arbitrage opportunity found")
    except Exception as e:
        print(f"✗ Slippage analysis failed: {str(e)}")
    
    # Test market volatility impact
    try:
        print("\n[Test 8] Market volatility analysis")
        volatility_levels = [0.01, 0.02, 0.05, 0.1]  # 1% to 10%
        for volatility in volatility_levels:
            print(f"\nTest volatility: {volatility*100}%")
            lst_tool.update_config({"market_volatility": volatility})
            result = await lst_tool.find_circular_arbitrage("ETH", 1.0)
            if "best_opportunity" in result:
                opp = result["best_opportunity"]
                print(f"Profit: {opp['profit']:.6f} ETH ({opp['profit_percent']:.2f}%)")
                print(f"Expected volatility range: ±{volatility*100}%")
            else:
                print("No arbitrage opportunity found")
    except Exception as e:
        print(f"✗ Volatility analysis failed: {str(e)}")

async def main():
    print("=== LST Arbitrage Tool Test ===")
    
    # Check environment variables
    api_key = os.environ.get("BITQUERY_API_KEY")
    client_id = os.environ.get("BITQUERY_CLIENT_ID")
    client_secret = os.environ.get("BITQUERY_CLIENT_SECRET")
    print("api_key", api_key)
    print("client_id", client_id)
    print("client_secret", client_secret)
    
    if not api_key or not client_id or not client_secret:
        print("Warning: Environment variables BITQUERY_API_KEY, BITQUERY_CLIENT_ID or BITQUERY_CLIENT_SECRET are not set")
        print("You can continue testing, but tests might fail if the base class uses these variables")
    
    # Run basic tests
    await run_basic_test()
    
    # Ask whether to run comprehensive tests
    choice = input("\nRun more comprehensive tests? (y/n): ")
    
    if choice.lower() == 'y':
        await run_comprehensive_test()
    
    print("\nTests completed!")

if __name__ == "__main__":
    asyncio.run(main())
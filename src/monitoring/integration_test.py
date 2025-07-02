"""
Integration Test Script
Run this to verify the enhanced monitoring integration works
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

async def test_enhanced_monitoring_integration():
    """FIXED: Test that all enhanced components are working"""
    
    print("🧪 Testing Enhanced Monitoring Integration (FIXED)...")
    print("=" * 50)
    
    # Test 1: Import test
    print("📦 Testing imports...")
    try:
        from src.monitoring.market_regime_detector import MarketRegimeDetector, MarketRegime
        from src.monitoring.decision_scorer import EnhancedDecisionScorer, DecisionContext, DecisionQuality
        from src.monitoring.agent_performance_monitor import PerformanceMonitor
        from src.monitoring.monitored_crew import MonitoredAutonomousTradingSystem
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Enhanced PerformanceMonitor initialization
    print("\n🏗️ Testing enhanced PerformanceMonitor...")
    try:
        monitor = PerformanceMonitor()
        
        # Check if enhanced components are present
        assert hasattr(monitor, 'regime_detector'), "Missing regime_detector"
        assert hasattr(monitor, 'decision_scorer'), "Missing decision_scorer"
        assert hasattr(monitor, 'price_history'), "Missing price_history"
        assert hasattr(monitor, 'update_price_data'), "Missing update_price_data method"
        
        print("✅ Enhanced PerformanceMonitor initialized with all components")
    except Exception as e:
        print(f"❌ PerformanceMonitor enhancement failed: {e}")
        return False
    
    # Test 3: Market regime detection
    print("\n📊 Testing market regime detection...")
    try:
        regime_detector = MarketRegimeDetector()
        
        # Test with minimal dummy data
        dummy_data = [
            {'timestamp': '2025-01-01 10:00:00', 'open': 1.1000, 'high': 1.1010, 'low': 1.0990, 'close': 1.1005, 'volume': 1000},
            {'timestamp': '2025-01-01 10:15:00', 'open': 1.1005, 'high': 1.1015, 'low': 1.0995, 'close': 1.1010, 'volume': 1200},
        ]
        
        regime_info = regime_detector.detect_current_regime(dummy_data, 1.1010)
        
        # Check expected structure
        required_keys = ['primary_regime', 'regime_strength', 'volatility_percentile']
        for key in required_keys:
            assert key in regime_info, f"Missing key: {key}"
        
        print(f"✅ Regime detection working: {regime_info['primary_regime']}")
        print(f"   Regime strength: {regime_info['regime_strength']:.1f}%")
    except Exception as e:
        print(f"❌ Regime detection failed: {e}")
        return False
    
    # Test 4: Enhanced decision scoring
    print("\n🎯 Testing enhanced decision scoring...")
    try:
        scorer = EnhancedDecisionScorer()
        
        # Create test context
        context = DecisionContext(
            market_regime='trending_bullish',
            volatility_level=0.5,
            session_time='london_session',
            recent_performance=70.0,
            consecutive_wins=2,
            consecutive_losses=0,
            time_pressure=0.3,
            complexity_score=0.4
        )
        
        # Test scoring
        result = scorer.score_decision(
            confidence=75.0,
            outcome_positive=True,
            outcome_value=50.0,
            context=context,
            execution_time_ms=1500.0
        )
        
        # Check result structure
        required_keys = ['overall_score', 'quality_rating', 'component_scores', 'recommendations']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        print(f"✅ Decision scoring working: {result['quality_rating']}")
        print(f"   Overall score: {result['overall_score']:.1f}/100")
    except Exception as e:
        print(f"❌ Decision scoring failed: {e}")
        return False
    
    # Test 5: Enhanced system integration - FIXED VERSION
    print("\n🤖 Testing enhanced system integration...")
    try:
        # Initialize enhanced system (NO async calls in __init__ now)
        system = MonitoredAutonomousTradingSystem()
        
        # Check if it has the enhanced monitor
        assert hasattr(system.performance_monitor, 'regime_detector'), "System missing enhanced monitor"
        
        # Test methods exist
        assert hasattr(system, 'update_market_data'), "Missing update_market_data method"
        assert hasattr(system, '_ensure_monitoring_started'), "Missing _ensure_monitoring_started method"
        
        # Test that monitoring can be started properly
        await system._ensure_monitoring_started()
        assert system.monitoring_started, "Monitoring not started"
        
        print("✅ Enhanced system integration successful")
        
        # Cleanup
        await system.shutdown_monitoring()
        
    except Exception as e:
        print(f"❌ System integration failed: {e}")
        return False
    
    # Test 6: Enhanced dashboard data
    print("\n📋 Testing enhanced dashboard data...")
    try:
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()  # Start monitoring properly
        
        # Get dashboard data
        dashboard_data = monitor.get_real_time_dashboard_data()
        
        # Should have basic structure
        assert 'timestamp' in dashboard_data, "Missing timestamp"
        assert 'monitoring_active' in dashboard_data, "Missing monitoring_active"
        
        print("✅ Enhanced dashboard data structure ready")
        
        # Cleanup
        await monitor.stop_monitoring()
        
    except Exception as e:
        print(f"❌ Enhanced dashboard test failed: {e}")
        return False
    
    print("\n🎉 ALL INTEGRATION TESTS PASSED!")
    print("=" * 50)
    print("✅ Your enhanced monitoring system is ready to use!")
    print("\n🚀 Next steps:")
    print("1. Use the FIXED MonitoredAutonomousTradingSystem")
    print("2. Start using it in your trading (monitoring auto-starts)")
    print("3. Let it collect data for at least 50 decisions")
    print("4. Review the enhanced metrics and recommendations")
    
    return True

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_enhanced_monitoring_integration())  # ✅ Properly awaited
    if not success:
        print("\n❌ Integration incomplete. Please review the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Integration successful! Ready for enhanced monitoring.")
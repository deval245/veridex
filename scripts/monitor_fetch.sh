#!/bin/bash

LOG_FILE="/Users/devalthakkar/Documents/veridex/data/fetch_20k_live.log"

echo "🔍 VERIDEX FETCH MONITOR"
echo "========================"
echo "Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "🔍 VERIDEX FETCH MONITOR - $(date '+%H:%M:%S')"
    echo "================================================================"
    
    if [ -f "$LOG_FILE" ]; then
        echo ""
        tail -30 "$LOG_FILE"
        echo ""
        echo "================================================================"
        
        # Check if completed
        if grep -q "DATASET FETCH COMPLETE" "$LOG_FILE"; then
            echo "✅ FETCH COMPLETED!"
            
            # Show final stats
            echo ""
            grep -A 20 "DATASET FETCH COMPLETE" "$LOG_FILE"
            break
        fi
        
        # Show current progress
        movies_collected=$(grep -o "[0-9]* movies collected" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" | head -1)
        if [ -n "$movies_collected" ]; then
            echo "📊 Current status: $movies_collected movies collected"
            progress=$((movies_collected * 100 / 20000))
            echo "📈 Progress: $progress% complete"
        fi
    else
        echo "❌ Log file not found. Process may not have started."
    fi
    
    echo ""
    echo "Next update in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done










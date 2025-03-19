#!/bin/bash
# QA Vision - All-in-one script for UI alignment detection

# Display help information
show_help() {
    echo "QA Vision - UI Alignment Detection Tool"
    echo "========================================"
    echo ""
    echo "Usage: ./run_qa_vision.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup         Setup the project and install dependencies"
    echo "  train         Train a LoRA adapter for UI alignment detection"
    echo "  analyze       Analyze UI screenshots for alignment issues"
    echo "  inference     Run inference on a single image"
    echo "  prepare       Prepare a dataset for training"
    echo "  convert       Convert a trained adapter to GGUF format"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_qa_vision.sh setup               # Setup the project"
    echo "  ./run_qa_vision.sh train               # Train with default settings"
    echo "  ./run_qa_vision.sh analyze --image /path/to/image.png   # Analyze an image"
    echo ""
    echo "For more options, run a command with --help, e.g.:"
    echo "  ./run_qa_vision.sh train --help"
}

# Ensure scripts are executable
chmod +x scripts/*.py *.sh

# If no arguments, show help
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Parse command
COMMAND=$1
shift  # Remove the command from the arguments

case $COMMAND in
    setup)
        ./setup_project.sh "$@"
        ;;
    train)
        python scripts/train_lora.py "$@"
        ;;
    analyze)
        python scripts/analyze_ui.py "$@"
        ;;
    inference)
        python scripts/inference.py "$@"
        ;;
    prepare)
        python scripts/prepare_dataset.py "$@"
        ;;
    convert)
        python scripts/convert_lora_to_gguf.py "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run './run_qa_vision.sh help' for usage information."
        exit 1
        ;;
esac
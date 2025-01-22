from utils.parser import create_cli_parser, validate_args
from utils.model_loader import ensure_hf_token, load_chat_pipeline, build_logits_processor_list
from utils.chatter import Chat

def main():
    ensure_hf_token()

    parser = create_cli_parser()
    args = parser.parse_args()

    validate_args(args)

    pipeline = load_chat_pipeline(args)

    print("Model loaded successfully!\n")
    print("Commands:\n")
    print(" - Just type a message and press Enter for a response.")
    print(" - 'regenerate' to regenerate the last assistant response.")
    print(" - 'regenerate N' to regenerate assistant response for turn N.")
    print(" - 'edit' to modify user input from last turn and regenerate subsequent turns.")
    print(" - 'edit N' to modify user input at turn N and regenerate subsequent turns.")
    print(" - 'revert N' to revert conversation to turn N (remove subsequent turns).")
    print(" - 'history' to print the current conversation.")
    print(" - 'set_params param1=val1 param2=val2' to update params on logit processors.")
    print(" - 'show_params' to show current params on logit processor.")
    print(" - 'clear analysis' to clear the analysis log.")
    print(" - 'FEWSHOT <input_text>' to format the input for fewshot learning.")
    print(" - 'exit' or 'quit' to end the session.\n")

    processors = build_logits_processor_list(args)

    chat = Chat(
        logging_enabled=args.log,
        system_prompt=args.system_prompt,
        fewshot=args.fewshot,
        fewshot_num=args.fewshot_num,
        logits_processor_spec=args.logits_processor,
        pipeline=pipeline,
        processors=processors,
        model_name=args.model
    )

    while(True):
        user_input = input("User: ").strip()
        if not user_input:
            continue

        # ------------------------------------------------
        # Handle special commands
        # ------------------------------------------------
        if user_input.lower() in ("exit", "quit"):
            print("Exiting the conversation. Goodbye!")
            break

        if user_input.startswith("revert "):
            parts = user_input.split()
            if len(parts) == 2:
                turn_id = int(parts[1])
                chat.revert_to(turn_id)
                continue
            elif len(parts) == 1:
                turn_id = chat.get_num_turns() - 1
                chat.revert_to(turn_id)
                continue

        if user_input.startswith("edit"):
            parts = user_input.split()
            if len(parts) == 2:
                turn_id = int(parts[1])
                chat.edit_turn(turn_id)
                continue
            elif len(parts) == 1:
                turn_id = chat.get_num_turns() - 1
                chat.edit_turn(turn_id)
                continue


        if user_input.startswith("regenerate"):
            parts = user_input.split()
            if len(parts) == 2:
                turn_id = int(parts[1])
                chat.regenerate(turn_id)
                continue
            elif len(parts) == 1:
                turn_id = chat.get_num_turns() - 1
                chat.regenerate(turn_id)
                continue

        if user_input.lower() == "history":
            chat.print_history()
            continue

        if user_input.startswith("set_params"):
            parts = user_input.split()
            if len(parts) > 1:
                for part in parts[1:]:
                    key, value = part.split("=")
                    chat.set_params(key, value)
            continue

        if user_input.startswith("show_params"):
            chat.show_params()
            continue

        if user_input.lower() == "clear analysis":
            if args.analysis:
                chat.clear_analysis()
            continue

        if user_input.startswith("FEWSHOT"):
            user_input = user_input[len("FEWSHOT"):].strip()
            user_input = chat.fewshot_format(user_input)
        
        # ------------------------------------------------
        # Typical user input
        # ------------------------------------------------
        chat.add_turn(user_input)
    
    chat.close_log()

if __name__ == "__main__":
    main()

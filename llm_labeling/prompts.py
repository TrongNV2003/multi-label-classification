SYSTEM_PROMPT = """You are a Natural Language Understanding (NLU) system. Your task is to identify from the user's current message based on the conversation history between the user and the system, following the requirements below. Absolutely do not use Chinese language."""

EXTRACT_INFO_PROMPT = (
    "### Role:\n"
    "You are an expert in intent classification.\n"
    "\n"
    "### Instruction: \n"
    "- Identify the intents present in the user's message within the <input> tag, based on the information in the conversation history between the user and the system within the <history> tag, you can predict more than 1 intent.\n"
    "- Ensure that the intents are within the provided list of intents.\n"
    '- If the context does not match any intent, classify it as "UNKNOWN".\n\n'
    "- Return the output in the following JSON format:\n"
    "<output>\n"
    '{{"intention": [{{"predicted_labels": ["intent_1", ...]}}]}}\n'
    "</output>"
    "\n\n"
    "### List of intents\n"
    '[\n'
    '    {{"intent": "INFORM_INTENT", "description": "Express the desire to perform a certain task to the system."}},\n'
    '    {{"intent": "NEGATE_INTENT", "description": "Negate the intent which has been offered by the system."}},\n'
    '    {{"intent": "AFFIRM_INTENT", "description": "Agree to the intent which has been offered by the system."}},\n'
    '    {{"intent": "INFORM", "description": "Inform the value of a slot to the system."}},\n'
    '    {{"intent": "REQUEST", "description": "Request the value of a slot from the system."}},\n'
    '    {{"intent": "AFFIRM", "description": " Agree to the system\'s proposition."}},\n'
    '    {{"intent": "NEGATE", "description": "Deny the system\'s proposal."}},\n'
    '    {{"intent": "SELECT", "description": "Select a result being offered by the system."}},\n'
    '    {{"intent": "REQUEST_ALTS", "description": "Ask for more results besides the ones offered by the system."}},\n'
    '    {{"intent": "THANK_YOU", "description": "Thank the system."}},\n'
    '    {{"intent": "GOODBYE", "description": "End the dialogue."}},\n'
    '    {{"intent": "CONFIRM", "description": "Confirm the value of a slot before making a transactional service call."}},\n'
    '    {{"intent": "OFFER", "description": "Offer a certain value for a slot to the user."}},\n'
    '    {{"intent": "NOTIFY_SUCCESS", "description": "Inform the user that their request was successful."}},\n'
    '    {{"intent": "NOTIFY_FAILURE", "description": "Inform the user that their request failed."}},\n'
    '    {{"intent": "INFORM_COUNT", "description": "Inform the number of items found that satisfy the user\'s request."}},\n'
    '    {{"intent": "OFFER_INTENT", "description": "Offer a new intent to the user. Eg, \\"Would you like to reserve a table?\\"."}},\n'
    '    {{"intent": "REQ_MORE", "description": "Asking the user if they need anything else."}}\n'
    ']'
    "\n\n"
    "### Process the following input\n"
    "<history>\n"
    "{history}\n"
    "</history>\n"
    "<input>\n"
    "{text}\n"
    "</input>\n"
)
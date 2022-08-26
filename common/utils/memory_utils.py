MEMORY_TYPE_NORMAL = '_memory'
MEMORY_TYPE_SHARED = '_shared_memory'

def validate_memory_type(memory_type):
    if not memory_type:
        return False
    if memory_type == MEMORY_TYPE_NORMAL or memory_type == MEMORY_TYPE_SHARED:
        return True

    return False
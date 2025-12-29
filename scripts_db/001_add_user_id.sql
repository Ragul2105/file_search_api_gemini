-- Migration: Add user_id to conversations and messages tables
-- Run this after truncating existing tables

-- Step 1: Truncate existing data (user confirmed this is OK)
TRUNCATE TABLE messages CASCADE;
TRUNCATE TABLE conversations CASCADE;

-- Step 2: Add user_id column to conversations
ALTER TABLE conversations 
ADD COLUMN user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE;

-- Step 3: Add user_id column to messages
ALTER TABLE messages 
ADD COLUMN user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE;

-- Step 4: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_user ON messages(conversation_id, user_id);

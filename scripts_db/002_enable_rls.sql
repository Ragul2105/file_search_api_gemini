-- Migration: Enable Row Level Security (RLS) on tables
-- This ensures users can only access their own data

-- Enable RLS on conversations
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- Enable RLS on messages
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- CONVERSATIONS POLICIES
-- =====================================================

-- Users can only view their own conversations
CREATE POLICY "Users can view own conversations"
ON conversations FOR SELECT
TO authenticated
USING ((SELECT auth.uid()) = user_id);

-- Users can only create their own conversations
CREATE POLICY "Users can create own conversations"
ON conversations FOR INSERT
TO authenticated
WITH CHECK ((SELECT auth.uid()) = user_id);

-- Users can only update their own conversations
CREATE POLICY "Users can update own conversations"
ON conversations FOR UPDATE
TO authenticated
USING ((SELECT auth.uid()) = user_id)
WITH CHECK ((SELECT auth.uid()) = user_id);

-- Users can only delete their own conversations
CREATE POLICY "Users can delete own conversations"
ON conversations FOR DELETE
TO authenticated
USING ((SELECT auth.uid()) = user_id);

-- =====================================================
-- MESSAGES POLICIES
-- =====================================================

-- Users can view messages from their own conversations
CREATE POLICY "Users can view own messages"
ON messages FOR SELECT
TO authenticated
USING ((SELECT auth.uid()) = user_id);

-- Users can create messages in their own conversations
CREATE POLICY "Users can create own messages"
ON messages FOR INSERT
TO authenticated
WITH CHECK ((SELECT auth.uid()) = user_id);

-- Users can delete their own messages
CREATE POLICY "Users can delete own messages"
ON messages FOR DELETE
TO authenticated
USING ((SELECT auth.uid()) = user_id);

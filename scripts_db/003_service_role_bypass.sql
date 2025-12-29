-- Migration: Allow service role to bypass RLS
-- This is needed because FastAPI uses the service role key
-- The backend handles user_id filtering in queries

-- Grant service role full access (bypasses RLS)
-- Note: In Supabase, service_role automatically bypasses RLS
-- But we need to ensure the backend uses the service_role key, not anon key

-- For development/testing: Allow anon role to also access (remove in production)
-- CREATE POLICY "Anon access for dev" ON conversations FOR ALL TO anon USING (true);
-- CREATE POLICY "Anon access for dev" ON messages FOR ALL TO anon USING (true);

-- IMPORTANT: The backend should use SUPABASE_SERVICE_ROLE_KEY instead of SUPABASE_KEY
-- This key bypasses RLS, and the backend will handle user_id filtering manually

-- Verify RLS is enabled
SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';

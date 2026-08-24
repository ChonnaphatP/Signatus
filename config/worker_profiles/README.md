# Worker profiles

The Signatus Worker Profile tool stores reusable profile JSON files here by default.
Each v1 profile contains exactly a worker ID, display name, and complete face-image
data URI. It does not store an SFace embedding.

During Wo.No. Create or Edit, Core sends the profile's face image to the AI Service
to generate a strict 128-value SFace embedding. Only the worker ID, name, and newly
generated embedding are copied into the Wo.No. file, so runtime authorization does
not depend on this directory.

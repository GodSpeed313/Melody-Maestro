-- CreateTable
CREATE TABLE "Melody" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT,
    "bpm" INTEGER NOT NULL DEFAULT 120,
    "key" TEXT NOT NULL,
    "scale" TEXT NOT NULL,
    "notes" JSONB NOT NULL,
    "tags" TEXT,
    "isFavorite" BOOLEAN NOT NULL DEFAULT false
);

import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

async function main() {
  const melody = await prisma.melody.create({
    data: {
      name: 'Detroit Night Drive',
      bpm: 144,
      key: 'E',
      scale: 'Minor',
      notes: '[64, 67, 71, 74]',
      tags: 'Dark, Gritty, Cardo Style',
    },
  })
  console.log('Created melody:', melody)
}

main()
  .catch((e) => console.error(e))
  .finally(async () => await prisma.$disconnect())
      CHARACTER  datafile*100,argv*100,IN_STR*10
      REAL inweights(10,64), aiweights(50, 64, 64), outweights(64,1)
      REAL inb(64), aib(50, 64), outb(1)


      do i = 1, iargc()
        if (i==1) call getarg(i, argv)
      end do
      READ(argv, *) datafile
      print*, "infile = ",datafile

      call readData(datafile, inweights, aiweights, outweights, inb, aib, outb, 10,50,64)
      print *, "JUST A TEST"
      print *, "WEIGHTS of THE ENCODING BLOCK"
      do i = 1, 10
          print *, inweights(i,:)
          print *,""
      end do
      print *, "BIASes of THE ENCODING BLOCK"
      print *,""
      print *,""
      print *, inb(:)
      end

      SUBROUTINE readData(filename, inw, aiw, outw, inb, aib, outb, in_num, block_num, ai_num)
      INTEGER in_num, block_num, ai_num
      REAL inw(in_num,ai_num), aiw(block_num, ai_num, ai_num), outw(ai_num,1)      
      REAL inb(ai_num), aib(block_num, ai_num), outb(1)
      CHARACTER  filename*100
      CHARACTER  LINE*1200
      CHARACTER  tLINE*1200
      CHARACTER  ch, lastEntry*100
      INTEGER    b, l, c

      open (unit=1, file=filename, form='formatted')      
      !parse the encoding block
      do l = 1,in_num
        read(1, "(A)", iostat=io) LINE
        tLINE= LINE(3:)
        READ(tLINE, *) inw(l,1) 
        if(tLINE(10:10) == 'e') then
                tLINE= tLINE(14:)
        else if(tLINE(11:11) == 'e')then
                tLINE= tLINE(15:)
        else if(tLINE(12:12) == 'e') then
                tLINE= tLINE(16:)
        end if
        do c = 2,ai_num-1
           READ(tLINE, *) inw(l,c) 
           if(tLINE(11:11) == 'e') then
                tLINE= tLINE(15:)
           else if(tLINE(12:12) == 'e') then
                tLINE= tLINE(16:)
           else if(tLINE(13:13) == 'e') then
                tLINE= tLINE(17:)
           end if
        end do
        READ(tLINE, *) lastEntry
        !if (lastEntry(15:15) == ']') lastEntry=lastEntry(1:14)        
        if (lastEntry(14:14) == ']') then
           lastEntry=lastEntry(1:13)
        else if (lastEntry(15:15) == ']') then 
           lastEntry=lastEntry(1:14)
        else if (lastEntry(16:16) == ']') then
           lastEntry=lastEntry(1:15)
        end if
        READ(lastEntry, *) inw(l,ai_num)
      end do        

      !skip 2 lines
      read(1, "(A)", iostat=io) LINE
      read(1, "(A)", iostat=io) LINE

      !parse the bias
      read(1, "(A)", iostat=io) LINE
      tLINE= LINE(2:)
      READ(tLINE, *) inb(1) 
      if(tLINE(10:10) == 'e') then
                tLINE= tLINE(14:)
      else if(tLINE(11:11) == 'e')then
                tLINE= tLINE(15:)
      else if(tLINE(12:12) == 'e') then
                tLINE= tLINE(16:)
      end if
      do c = 2,ai_num-1
         READ(tLINE, *) inb(c) 
         if(tLINE(11:11) == 'e') then
                tLINE= tLINE(15:)
         else if(tLINE(12:12) == 'e') then
                tLINE= tLINE(16:)
         else if(tLINE(13:13) == 'e') then
                tLINE= tLINE(17:)
         end if
      end do
      READ(tLINE, *) lastEntry
      if (lastEntry(14:14) == ']') then
           lastEntry=lastEntry(1:13)
      else if (lastEntry(15:15) == ']') then 
           lastEntry=lastEntry(1:14)
      else if (lastEntry(16:16) == ']') then
           lastEntry=lastEntry(1:15)
      end if
      READ(lastEntry, *) inb(ai_num)

      !skip 3 lines
      read(1, "(A)", iostat=io) LINE
      read(1, "(A)", iostat=io) LINE
      read(1, "(A)", iostat=io) LINE

      !parse the compute block
      do b = 1,block_num
        do l = 1,ai_num
           read(1, "(A)", iostat=io) LINE
           tLINE= LINE(3:)
           READ(tLINE, *) aiw(b,l,1)
           if(tLINE(10:10) == 'e') then
                tLINE= tLINE(14:)
           else if(tLINE(11:11) == 'e')then
                tLINE= tLINE(15:)
           else if(tLINE(12:12) == 'e') then
                tLINE= tLINE(16:)
           end if
           do c = 2,ai_num-1
               READ(tLINE, *) aiw(b,l,c)
               if(tLINE(11:11) == 'e') then
                   tLINE= tLINE(15:)
               else if(tLINE(12:12) == 'e')then
                   tLINE= tLINE(16:)
               else if(tLINE(13:13) == 'e') then
                   tLINE= tLINE(17:)
           end if
           end do
           READ(tLINE, *) lastEntry
           if (lastEntry(14:14) == ']') then
                   lastEntry=lastEntry(1:13)
           else if (lastEntry(15:15) == ']') then 
                   lastEntry=lastEntry(1:14)
           else if (lastEntry(16:16) == ']') then
                   lastEntry=lastEntry(1:15)
           end if
           READ(lastEntry, *) aiw(b,l,ai_num)
        end do
        !skip 2 lines
        read(1, "(A)", iostat=io) LINE
        read(1, "(A)", iostat=io) LINE

        !parse the bias
        read(1, "(A)", iostat=io) LINE
        tLINE= LINE(2:)
        READ(tLINE, *) aib(b,1) 
        if(tLINE(10:10) == 'e') then
                tLINE= tLINE(14:)
        else if(tLINE(11:11) == 'e')then
                tLINE= tLINE(15:)
        else if(tLINE(12:12) == 'e') then
                tLINE= tLINE(16:)
        end if

        do c = 2,ai_num-1
           READ(tLINE, *) aib(b,c) 
           if(tLINE(10:10) == 'e')then
                   tLINE= tLINE(14:)
           else if(tLINE(11:11) == 'e')then
                   tLINE= tLINE(15:)
           else if(tLINE(12:12) == 'e') then
                   tLINE= tLINE(16:)
           else if(tLINE(13:13) == 'e') then
                   tLINE= tLINE(17:)
           end if
        end do
        READ(tLINE, *) lastEntry
        if (lastEntry(14:14) == ']') then
                lastEntry=lastEntry(1:13)        
        else if (lastEntry(15:15) == ']') then
                lastEntry=lastEntry(1:14)        
        else if (lastEntry(16:16) == ']') then
                lastEntry=lastEntry(1:15)        
        end if
        READ(lastEntry, *) aib(b,ai_num)

        !skip 3 lines
        read(1, "(A)", iostat=io) LINE
        read(1, "(A)", iostat=io) LINE
        read(1, "(A)", iostat=io) LINE
      end do

      !parse the output block
      do l = 1,ai_num
         read(1, "(A)", iostat=io) LINE
         tLINE= LINE(3:)
         if (tLINE(14:14) == ']') then 
                 tLINE=tLINE(1:13)
         else if (tLINE(15:15) == ']') then 
                 tLINE=tLINE(1:14)
         else if (tLINE(16:16) == ']') then 
                 tLINE=tLINE(1:15)
         end if
         READ(tLINE, *) outw(l,1)
         print *, outw(l,1)
      end do

      !skip 2 lines
      read(1, "(A)", iostat=io) LINE
      read(1, "(A)", iostat=io) LINE

      !parse the bias
      read(1, "(A)", iostat=io) LINE
      tLINE= LINE(2:)
      if (tLINE(12:12) == ']') then 
         tLINE=tLINE(1:11)
      else if (tLINE(11:11) == ']') then 
         tLINE=tLINE(1:10)
      else if (tLINE(10:10) == ']') then 
         tLINE=tLINE(1:9)
      end if
      READ(tLINE, *) outb(1)

      CLOSE (1, STATUS='KEEP')
      return              

      end
